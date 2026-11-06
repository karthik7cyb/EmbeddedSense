# Batch Processor

This guide explains what the batch processor does, what each constant means, and how a run works from start to finish.

**Code:** [`app/ai/batchs/batch_processing.py`](../app/ai/batchs/batch_processing.py)
**Used by:** [`app/ai/rag/embed.py`](../app/ai/rag/embed.py)

---

## Why it exists

During ingestion, a document is split into chunks and every chunk needs an embedding. A large document can produce thousands of chunks.

Embeddings come from Gemini's `gemini-embedding-001`. The free tier limits how much you can send per minute, and sending too much returns `429` errors that make ingestion fail.

The batch processor solves this by doing four jobs:

1. Splits the chunks into smaller groups (batches).
2. Limits how many batches run at the same time.
3. Waits when the per-minute limits are reached.
4. Retries batches that fail.

It knows nothing about embeddings. It takes a list and a function, so you can reuse it for other work later.

---

## The analogy: a delivery company

Picture a company that has to deliver a warehouse full of parcels across a toll bridge.

| Delivery company | In the code | Setting |
|---|---|---|
| A parcel | One chunk of text | |
| A van full of parcels | One batch (one API request) | `batch_size` |
| Vans allowed on the road at once | Batches running at the same time | `max_concurrency` |
| Toll gate: vans allowed per minute | Requests per minute | `requests_per_minute` |
| Bridge: total weight allowed per minute | Tokens per minute | `tokens_per_minute` |
| The minute the bridge staff count over | The rate limit window | `RATE_LIMIT_WINDOW` |
| Guessing a parcel's weight from its size | Estimating tokens from characters | `CHARS_PER_TOKEN` |
| Delivery attempts before giving up | Attempts per batch | `max_retries` |
| Wait after the first failed delivery | First retry wait | `BASE_RETRY_DELAY` |
| Longest wait between attempts | Retry wait cap | `MAX_RETRY_DELAY` |

The sections below use this analogy for each constant.

---

## The constants

There are two groups of constants at the top of `batch_processing.py`.

### Group 1: the settings presets

Each preset is a dictionary you pass to `BatchProcess`.

```python
BatchProcess(**GEMINI_FREE_TIER_SETTINGS)
```

| Setting | `LOCAL_EMBEDDING_SETTINGS` (not in use) | `GEMINI_FREE_TIER_SETTINGS` (in use) |
|---|---|---|
| `batch_size` | 32 | 20 |
| `max_concurrency` | 1 | 2 |
| `requests_per_minute` | `None` (no limit) | 80 |
| `tokens_per_minute` | `None` (no limit) | 25,000 |
| `max_retries` | 1 | 5 |

#### `batch_size`: how many items go in one request

**Analogy:** how many parcels fit in one van.

**Example:** 70 chunks with `batch_size = 32` makes 3 batches of 32, 32 and 6 chunks.

**Trade-off:**
- Larger batches mean fewer requests, but one failure means resending more chunks.
- Smaller batches mean more requests, which uses up your requests-per-minute and requests-per-day limits faster.

**Why 32 and 20?** 32 is a common batch size for local models, and it's the default in `sentence-transformers`. 20 keeps a Gemini batch at about 9,000 tokens, which is well under the per-minute limit. See [Tuning for the free tier](#tuning-for-the-free-tier) for how to choose this value.

#### `max_concurrency`: how many batches run at the same time

**Analogy:** how many vans can be on the road at once. The rest wait in the yard until a van returns.

**Example:** with 6 batches and `max_concurrency = 2`:

```
Time →
Slot 1: [batch 1][batch 3][batch 5]
Slot 2: [batch 2][batch 4][batch 6]
```

**Why 1 for local?** The model runs on your own CPU, so running batches side by side wouldn't make it any faster.

**Why 2 for Gemini?** The work is waiting on a network response, so a small amount of overlap saves time. On the free tier the per-minute limits are what really slow you down, so a higher number wouldn't help much.

**Concurrency is not the same as a rate limit.** `max_concurrency` limits how many run *right now*. `requests_per_minute` limits how many start *per minute*. If each request takes 1 second, then `max_concurrency = 2` could still send 120 requests a minute, which is why both settings exist.

#### `requests_per_minute`: how many requests can start per minute

**Analogy:** the toll gate lets through a fixed number of vans each minute, however empty they are.

**Example:** with `requests_per_minute = 80`, the 81st request within 60 seconds waits until the oldest request is more than 60 seconds old.

**Why 80?** The reported free-tier limit is 100. Staying at 80 leaves a safety margin. `None` turns the limit off.

Every attempt counts, including retries, because a failed request still uses your quota on the provider's side.

#### `tokens_per_minute`: how much text can be sent per minute

**Analogy:** the bridge allows a total weight per minute. A few heavy vans can reach the limit long before the toll gate's van count does.

**Example:** a full 300-word chunk is about 1,800 characters, or 450 tokens. A batch of 20 is about 9,000 tokens. With a limit of 25,000:

| Batch | Tokens already used | Tokens after this batch | Result |
|---|---|---|---|
| 1 | 0 | 9,000 | Sent |
| 2 | 9,000 | 18,000 | Sent |
| 3 | 18,000 | 27,000 | Waits |

**Why 25,000?** The reported free-tier limit is 30,000. Token counts here are only estimates, so the margin covers the difference.

**This is the limit you will hit first on Gemini.** With full-size chunks you only reach 2 requests a minute, nowhere near the limit of 80.

#### `max_retries`: total attempts per batch

**Analogy:** how many times the driver tries to deliver before returning the parcel as failed.

**This counts total attempts, not extra attempts:**

| `max_retries` | First try | Retries |
|---|---|---|
| 1 | 1 | 0 (a failure is final) |
| 3 | 1 | 2 |
| 5 | 1 | 4 |

**Why 1 for local?** A local failure, such as running out of memory or a bad input, will most likely fail again the same way.

**Why 5 for Gemini?** Network errors and `429`s are often temporary.

**Keep this at 1 or higher.** The value isn't validated, and `0` would make `run` crash with a `TypeError`, because no attempt ever runs.

### Group 2: shared constants

These apply to every `BatchProcess`. They aren't in the presets, so you change them by editing the file.

```python
BASE_RETRY_DELAY = 5.0
MAX_RETRY_DELAY = 60.0
RATE_LIMIT_WINDOW = 60.0
CHARS_PER_TOKEN = 4
```

#### `BASE_RETRY_DELAY` and `MAX_RETRY_DELAY`: how long to wait before retrying

**Analogy:** if nobody answers the door, the driver waits a little before trying again, then longer each time. Drivers also pick a random wait, so they don't all come back at the same moment.

The longest possible wait doubles after each failure and stops at the cap. The actual wait is a random time between 0 and that value. This is called **exponential backoff with full jitter**.

```python
backoff = min(MAX_RETRY_DELAY, BASE_RETRY_DELAY * (2 ** (attempt - 1)))
sleep_time = random.uniform(0, backoff)
```

| Failed attempt | Longest possible wait | Example actual wait |
|---|---|---|
| 1 | 5s | 3.2s |
| 2 | 10s | 7.8s |
| 3 | 20s | 4.1s |
| 4 | 40s | 26.5s |
| 5 | Final with `max_retries = 5`, so the error is raised | |

With `max_retries = 5` the 60-second cap never comes into play. It only matters from the 5th failure onward (5 × 2⁴ = 80, capped to 60), which needs `max_retries` of 6 or more.

**Why random?** If two batches fail at the same moment and both wait exactly 5 seconds, they retry together and can fail together again. Random waits spread them out.

#### `RATE_LIMIT_WINDOW`: the length of the "minute"

**Analogy:** the period the bridge staff count over.

The limits use a **sliding window**. It doesn't reset at the top of each minute. At any moment, the code looks back over the last 60 seconds and counts what was sent.

**Example:** with a limit of 2 requests per window:

```
t = 0s    Request A sent        (A in window, 1 used)
t = 10s   Request B sent        (A, B in window, 2 used)
t = 20s   Request C must wait   (limit reached)
t = 60s   A is now 60s old and drops out, so C is sent (B, C in window)
```

The window can also be changed for one instance with the `window_seconds` argument. The tests use this to run with a very short window, so they don't have to wait a real minute.

#### `CHARS_PER_TOKEN`: the token estimate

**Analogy:** the depot guesses a parcel's weight from its size rather than weighing each one.

Counting tokens exactly would need the provider's tokenizer or an extra API call. English text averages about 4 characters per token, so this estimate is close enough.

**Example:** a batch with 36,000 characters in total is estimated at `36,000 // 4 = 9,000` tokens.

Estimates can be off, which is one reason `tokens_per_minute` is set below the real limit.

---

## How it works

Everything starts with one call:

```python
results = await batch_processor.run(items, func)
```

- `items` is the full list of work, such as chunk texts.
- `func` is an async function that takes one batch and returns one result per item.

### The flow

```
pipeline.py: embedding.embed_chunks(chunk_texts)
                        │
                        ▼
embed.py:    batch_processor.run(chunks, self._embed_batch)
                        │
                        ▼
1. create_batches       Split the items into batches
                        │
                        ▼
2. Create one task      All tasks start, but the semaphore lets
   per batch            only max_concurrency of them proceed
                        │
                        ▼
3. _process_batch       For each batch:
                          a. Wait for a free slot (semaphore)
                          b. Stop here if another batch has failed for good
                          c. Wait for rate limit room
                          d. Call func(batch)
                          e. Check the result count matches
                          f. On failure, wait and go back to step c
                        │
                        ▼
4. asyncio.gather       Wait for all tasks and collect results in order
                        │
                        ▼
5. Flatten              [[a, b], [c, d]] becomes [a, b, c, d]
```

### Step 1: `create_batches`

Slices the list into pieces of `batch_size`. The last one can be smaller.

### Step 2: `run` starts the tasks

Creates one task per batch, plus the two objects they share:

- A **semaphore**, which is the set of `max_concurrency` slots. A batch has to hold a slot to run.
- A **stop signal** (`asyncio.Event`), which tells waiting batches that another batch has failed for good.

### Step 3: `_wait_for_rate_limit`

The limiter remembers each request as `(timestamp, tokens)` in a queue, `self._window`. Before each attempt it does the following:

1. Removes entries older than the window.
2. Checks there's room for one more request and for this batch's tokens.
3. If there's room, it records the request and continues.
4. If not, it sleeps until the oldest entry expires, then checks again.

An `asyncio.Lock` allows only one batch to check at a time. Without it, two batches could both see the same free quota and both go ahead.

**Oversized batches:** if a single batch is bigger than `tokens_per_minute`, it could never fit. The code logs a warning and lets it through once the window is empty, so it can't wait forever. If you see this warning, lower `batch_size`, because the provider will probably reject that request.

### Step 4: `_process_batch`

Each attempt waits on the limiter, calls `func(batch)`, and checks that the result count equals the batch size.

The count check matters because of how results get matched up. Embeddings are matched to chunks purely by position, so a missing result would attach the wrong embedding to every chunk after it.

**Any exception triggers a retry**, including permanent ones like an invalid API key. Those use up all their attempts before failing.

### Step 5: `asyncio.gather` keeps the order

Batches can finish in any order, but `gather` returns results in the order the tasks were created. The first embedding therefore always belongs to the first chunk.

```
Finish order:   batch 2, batch 3, batch 1
Result order:   batch 1, batch 2, batch 3
```

### What happens when a batch fails for good

1. The batch sets the stop signal and raises its error.
2. Batches still waiting for a slot see the signal and exit without sending anything.
3. `run` cancels the remaining tasks and raises the error.
4. `data_ingestion` in `pipeline.py` catches it and marks the file as `failed`.
5. The next file in the folder is processed as normal.

Nothing from the failed file is stored, because chunks are only saved after all embeddings succeed. Its hash isn't in the database either, so the next server start tries the whole file again from the beginning.

---

## Worked example: 100 chunks on the Gemini preset

Suppose all 100 chunks are full size (450 tokens each).

**Batches:** 100 / 20 = 5 batches of about 9,000 tokens each.

| Time | What happens | Tokens in the last 60s |
|---|---|---|
| 0s | Batches 1 and 2 are sent | 18,000 |
| 0s | Batch 3 would reach 27,000, so it waits | 18,000 |
| 60s | Batches 1 and 2 expire; batches 3 and 4 are sent | 18,000 |
| 60s | Batch 5 would reach 27,000, so it waits | 18,000 |
| 120s | Batches 3 and 4 expire; batch 5 is sent | 9,000 |

**Total:** about 2 minutes, plus the time the requests themselves take.

**Throughput:** 2 batches × 20 chunks = about **40 chunks per minute** for full-size chunks. At this rate a 2,000-chunk document takes about 50 minutes.

---

## Reading the logs

The logs for a normal run look like this (the numbers are illustrative):

```
Batch processing started: 70 items in 4 batches (batch_size=20, max_concurrency=2)
Batch 1/4 done (20 items)
Batch 2/4 done (20 items)
Rate limit reached, waiting 58.9s
Batch 3/4 done (20 items)
Batch 4/4 done (10 items)
Batch processing finished: 70 items in 61.8s
```

Other messages you may see:

| Log message | Meaning | What to do |
|---|---|---|
| `Rate limit reached, waiting 42.3s` | The limiter is holding back a batch to stay under the limit | Nothing, this is normal |
| `Batch 2/6 attempt 1/5 failed: ... Retrying in 3.2s` | A request failed and will be retried | Nothing, unless it keeps happening |
| `Batch 2/6 failed after 5 attempts: ...` | The batch failed for good, so the file is marked `failed` | Read the error at the end of the line |
| `Batch of ~30000 tokens is over the 25000 tokens/min limit...` | One batch is larger than the whole limit | Lower `batch_size` |

---

## Tuning for the free tier

Reported free-tier limits for `gemini-embedding-001` are **100 requests/min, 30,000 tokens/min and 1,000 requests/day**. These numbers come from a [Google staff reply on the forum](https://discuss.ai.google.dev/t/hitting-rate-limit-on-gemini-batch-api-for-gemini-embedding-001/107091), not an official table, and they may have changed. Check your real limits in Google AI Studio. The [official rate limits page](https://ai.google.dev/gemini-api/docs/rate-limits) explains how they work.

**How to choose `batch_size`:** only whole batches fit in the window, so a batch size that uses the token limit closely gets the most through.

| `batch_size` | Tokens per batch | Batches per minute | Chunks per minute | Chunks per day at 1,000 requests |
|---|---|---|---|---|
| 10 | 4,500 | 5 | 50 | 10,000 |
| 20 (current) | 9,000 | 2 | 40 | 20,000 |
| 27 | 12,150 | 2 | 54 | 27,000 |
| 50 | 22,500 | 1 | 50 | 50,000 |

These figures assume full-size chunks of 450 tokens at `tokens_per_minute = 25,000`.

**What to change for each problem:**

| Problem | Change |
|---|---|
| Still getting `429`s | Lower `tokens_per_minute` and `requests_per_minute` |
| Hitting the daily request limit | Raise `batch_size` |
| "Over the tokens/min limit" warning | Lower `batch_size` |
| Lots of work is lost when a batch fails | Lower `batch_size` |
| You've moved to a paid tier | Raise the limits and `max_concurrency` |

**Use normal embedding requests with a list of texts.** Don't use Gemini's separate Batch API for this. It's [not available on the free tier](https://ai.google.dev/gemini-api/docs/pricing).

---

## Using it

### With a preset

```python
from app.ai.batchs.batch_processing import BatchProcess, GEMINI_FREE_TIER_SETTINGS

processor = BatchProcess(**GEMINI_FREE_TIER_SETTINGS)
```

### With your own values

```python
processor = BatchProcess(
    batch_size = 50,
    max_concurrency = 1,
    requests_per_minute = 60,
    tokens_per_minute = 25_000,
    max_retries = 5,
)
```

### With any kind of work

`func` must be async and must return exactly one result per input item.

```python
async def shout_batch(batch: list[str]) -> list[str]:
    return [text.upper() for text in batch]

results = await processor.run(["a", "b", "c"], shout_batch)
# ["A", "B", "C"]
```

### How embeddings use it

`app/ai/rag/embed.py` creates one processor with `GEMINI_FREE_TIER_SETTINGS`. `embed_chunks` sends chunks through it, and `_embed_batch` is the function it calls for each batch. Everything goes through that one processor, so files ingested one after another share the same per-minute limits.

**If you change the embedding model,** delete the `chroma_db/` folder so everything is re-ingested. Vectors from different models can't be mixed, and the hash check would otherwise skip files that are already stored.

---

## Known limitations

| Limitation | What it means |
|---|---|
| **No resume within a file** | If one batch fails for good, the whole file is redone on the next start |
| **Daily limit isn't tracked** | Going over 1,000 requests/day gives `429`s and the file is marked `failed`. Tracking this would need a counter that survives restarts |
| **Limiter memory resets on restart** | A restart forgets recent requests, so one soon after a large run could go over the provider's limit and need retries |
| **Queries aren't counted** | `embed_query` skips the batch processor, so your questions use quota without being counted |
| **Startup waits for ingestion** | The server only accepts requests after ingestion finishes, which can take a long time for large documents. Moving ingestion to a background task would fix this |
| **Permanent errors are retried** | An invalid API key uses up all attempts before failing |
| **"Limit: 0" errors** | [Some free-tier users report](https://discuss.ai.google.dev/t/constant-429-resource-exhausted-error-after-switching-to-the-more-recent-text-embedding-model-gemini-embedding-001-on-free-tier/107614) a `429` showing a quota of 0. Batching can't fix this, so check your quota in AI Studio |

---

## Quick reference

| Name | Where | Value | One-line meaning |
|---|---|---|---|
| `batch_size` | Preset | 32 / 20 | Items per request |
| `max_concurrency` | Preset | 1 / 2 | Batches running at once |
| `requests_per_minute` | Preset | None / 80 | Requests that can start per window |
| `tokens_per_minute` | Preset | None / 25,000 | Estimated tokens per window |
| `max_retries` | Preset | 1 / 5 | Total attempts per batch |
| `BASE_RETRY_DELAY` | File constant | 5.0s | Longest wait after the first failure |
| `MAX_RETRY_DELAY` | File constant | 60.0s | Cap on the retry wait |
| `RATE_LIMIT_WINDOW` | File constant | 60.0s | Length of the sliding window |
| `CHARS_PER_TOKEN` | File constant | 4 | Characters per estimated token |

Preset values are shown as local / Gemini.
