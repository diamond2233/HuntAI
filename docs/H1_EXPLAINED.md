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

**How to test it by hand:** `scripts/benchmark.py` has a `--workers` flag
exactly for this. Run the same jobs twice, once sequentially and once in
parallel:

```
python scripts/benchmark.py h1b_benchmark.json --workers 1 --runs 3
python scripts/benchmark.py h1b_benchmark.json --workers 8 --runs 3
```

Both commands collect and filter the jobs once, then repeat *only* the
matching stage 3 times at that worker count and save the median to
`results/h1b_benchmark.json`. Running both (same output filename) builds up
one file with both results side by side, and the second command prints a
comparison table. Our real run: 107.3s median at `--workers 1` vs 15.9s at
`--workers 8` -- about 6.7x faster, on the exact same 60 jobs both times.

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

One more detail: the OpenAI SDK itself already retries rate limits/5xx/
timeouts up to 2 times before our code even sees the error. Left alone,
that would stack with our own retries -- one failing call could end up
being attempted up to 12 times (4 of ours x 3 of the SDK's) instead of 4.
We create the client with `OpenAI(max_retries=0)` so the SDK never retries
on its own, leaving our retry loop as the only one in control.

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

## H1b: Why only 5 jobs survived the filter, and what we actually found

After H1 made matching faster, the next question was: why does the
benchmark only have 5 jobs to match in the first place, out of 265
collected? The first guess was "the location filter is too strict -- it
probably doesn't know 'Bengaluru' and 'Bangalore' are the same city."

**What we actually found, in order:**

1. We checked *why* each job was rejected (`scripts/benchmark.py` adds a
   `filter_rejections` count to its output, broken down by reason). The
   first real answer: 258 of 265 jobs were rejected for **role**, only 2 for
   location. The original guess was wrong -- the 5 original job boards
   (Razorpay, Groww, CRED, Meesho, Paytm) just don't post many engineering
   roles; most of their listings are sales, collections, product, and ops.
   The role filter itself was fine.
2. So instead of "fixing" a filter that wasn't broken, we added more real
   company boards (`scripts/check_boards.py` probes Greenhouse's and
   Lever's public APIs for a list of companies and reports what it finds --
   see below) to get enough real engineering jobs to benchmark with.
3. *Once* boards from global companies (Databricks, Okta, Twilio, Coinbase,
   GitLab, Airbnb) were added, the location guess turned out to be right
   after all, just not yet -- "Bengaluru, India" became the single biggest
   rejected-location reason (124 jobs), because those companies spell the
   city "Bengaluru", not "Bangalore". *That's* when we added the alias.
4. A second, bigger problem showed up at the same time: 215 of 250 jobs
   that passed the filter were things like "Remote - California" or
   "Remote, USA" -- not India at all. They were only getting through
   because the location filter just checked whether the word "remote"
   appeared *anywhere* in the location text, and "Remote - California"
   contains "remote". We fixed `matches_location()` so "Remote" only counts
   if the location doesn't clearly name somewhere else (it checks for
   "india", or accepts a bare "Remote" with nothing else attached).

The lesson: the filter funnel having few survivors can have more than one
cause, and they can be hiding behind each other -- the alias problem was
real, but it was invisible until there were enough non-Bangalore,
non-India-labeled jobs in the data for it to show up at all.

**How to see the rejection reasons yourself:**

```
python scripts/benchmark.py my_run.json
```

The printed JSON includes `"filter_rejections": {"role": N, "location": M}`,
and the console output below it lists the most common rejected titles and
rejected location strings with counts. If you ever suspect the filter is
too strict (or too loose) again, this is the first thing to run -- don't
guess, look at the real counts.

**How to check a new company and add it to the profile:**

```
python scripts/check_boards.py
```

This tries several slug guesses (plus a few known aliases for rebrands) for
a list of companies against both Greenhouse's and Lever's public APIs, and
prints what it finds -- including an India-specific job/engineering-job
count, and whether the board's own name actually matches the company you
were guessing (marked VERIFIED/UNVERIFIED, since a slug like "meta" could
belong to an unrelated company). Only add a hit to `config/profile.yaml` if
it's VERIFIED *and* has real India-based engineering jobs -- check
`results/check_boards_output.txt` for the full real output and reasoning
from the last run.

**Why we compare 1 worker vs 8 workers:**

H1's concurrency change (above) is only worth having if it actually makes
matching faster on *real* data, not just in a quick mocked test. Comparing
`--workers 1` against `--workers 8` on the exact same jobs (same filter
run, same 60 jobs, just matched twice) isolates the one thing that changed
-- how many OpenAI calls run at once -- from everything else (which jobs,
how many, network conditions). That's what makes "107.3s vs 15.9s" a fair
comparison instead of two numbers from two different situations.

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
