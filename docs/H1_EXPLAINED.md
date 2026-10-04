# H1: Making the matching stage more reliable

This explains the 3 changes made to `pipeline/matcher.py` on the
`h1-reliability` branch, in plain language. Read this alongside the real
numbers in `results/baseline.json` and `results/after_concurrency.json`.

## Change 1: Concurrency (run jobs in parallel)

**Before:** `match_jobs()` called OpenAI once per job, one job at a time, in
a simple loop:

```python
return [_match_job(client, model, job, profile) for job in jobs]
```

If you had 50 jobs and each OpenAI call took 3 seconds, that's 150 seconds,
back to back.

**Now:** `match_jobs()` hands all the jobs to a `ThreadPoolExecutor`, which
runs up to 8 of them (configurable) *at the same time* on separate threads.
It still returns one `JobMatch` per job, in the exact same order as the
input list -- it just doesn't wait for job 1 to finish before starting job 2.

**Why it's better:** an OpenAI API call spends almost all of its time
waiting for a response over the network -- the computer isn't actually busy
during that wait. Threads let us have many of those waits happening at once
instead of one after another. In our real benchmark, with 5 jobs, matching
time dropped from 15.08s to 5.56s.

**How to test it by hand:** run `python scripts/benchmark.py` twice -- once
on the `main` branch, once on `h1-reliability` -- and compare the `"match"`
time in the printed JSON. You can also open `pipeline/matcher.py` and change
`DEFAULT_MAX_WORKERS = 8` to `DEFAULT_MAX_WORKERS = 1`, re-run, and watch
matching time go back up to roughly what it was before.

## Change 2: Retry (try again on temporary errors)

**Before:** if the OpenAI call failed for *any* reason -- a typo in your
request, a rate limit, a server hiccup, anything -- `match_jobs()` gave up
immediately and raised an error.

**Now:** before giving up, we check *what kind* of error it was:

- Rate limit, timeout, connection error, or "server had a problem" (5xx) →
  these are usually temporary. Wait a bit and try again. We retry up to 3
  times, waiting longer each time (1s, then 2s, then 4s -- this is called
  "exponential backoff").
- Anything else (bad request, the model refused, the response didn't match
  the format we asked for) → retrying won't fix this, so we don't bother.
  It fails right away, same as before.

**Why it's better:** rate limits and brief server errors are common and
usually resolve themselves within a few seconds. Without retrying, a single
hiccup from OpenAI's side could fail a job that would have worked fine 2
seconds later. Retrying only the errors that are actually worth retrying
avoids wasting time on errors that will never succeed no matter how many
times you try.

**How to test it by hand:** the real OpenAI API rarely fails on demand, so
this is tested with a mock instead (see
`test_retries_on_temporary_error_then_succeeds` and
`test_retry_backoff_is_exponential` in `tests/test_matcher.py`). To see it
with your own eyes, run:

```
python -m pytest tests/test_matcher.py -k retry -v
```

Open those two tests and read them -- they make the mocked OpenAI client
fail with a timeout error twice, then succeed on the third try, and check
that the code waited 1s then 2s between attempts (using a mocked `sleep` so
the test itself doesn't actually take 3 seconds).

## Change 3: Failure isolation (one bad job doesn't ruin the batch)

**Before:** if even one job, out of however many you were matching, failed
after everything above, the *entire* `match_jobs()` call raised an
exception -- meaning you'd get zero results back, even if 49 out of 50 jobs
matched successfully.

**Now:** a job that still fails after all retries is logged (so you can see
it happened) and skipped -- the rest of the batch keeps going. You can pass
a `stats` dictionary into `match_jobs()` to find out how many jobs were
skipped:

```python
stats = {}
matches = match_jobs(jobs, profile, stats=stats)
print(stats["skipped_jobs"])  # e.g. 1
```

**Why it's better:** one flaky job (or one that OpenAI refuses for some
reason) shouldn't throw away every other job's results. If you're matching
50 jobs and 1 fails permanently, you should still get the other 49 -- not
nothing.

**How to test it by hand:**

```
python -m pytest tests/test_matcher.py -k "skipped or isolation or block" -v
```

Read `test_one_failing_job_does_not_block_the_others` -- it makes one job
always fail and another always succeed, and checks that the successful job
still comes back, with `stats["skipped_jobs"] == 1`.

## What stayed the same

- Still exactly one OpenAI call per job (no batching of multiple jobs into
  one request).
- Still the same `JobMatch` structure coming out.
- `match_node` in `pipeline/graph.py` didn't need to change at all --
  `max_workers` and `stats` are optional, so existing callers that don't
  pass them behave the same as before (just faster, and now resilient to
  single-job failures).

## Quick check: 3 questions

1. If OpenAI returns a "rate limit exceeded" error for one job, roughly how
   long will the code wait before the *second* retry attempt, and why does
   each wait get longer instead of staying the same?
2. Suppose you're matching 10 jobs, `max_workers` is 8, and job #3
   permanently fails (not a temporary error) even after retries. How many
   `JobMatch` objects does `match_jobs()` return, and what does
   `stats["skipped_jobs"]` equal?
3. Why doesn't a `BadRequestError` (e.g. a malformed request) get retried
   the way a `RateLimitError` does?
