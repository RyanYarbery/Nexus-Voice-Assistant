# Parked work

Ideas agreed on but not started. Pick one up from here.

## 1. Web search that works like the ChatGPT app (parked 2026-11-14)

**Ryan's words (2026-11-14):** "That would be fantastic but I would need it to work in the same way
that the chat gpt app uses the internet so it would be as effective. Let's just save that as
something to be worked on. We'll come back to this later."

**Why:** the ChatGPT app answers current questions better because it searches the web. Nexus can't:
its only tools are location, date/time, weather and timers (`echo.py` around line 5358), so it
answers news, sports, prices and opening hours from the model's memory.

**Driving the ChatGPT app itself was ruled out:** it has no supported way for another program to send
it questions and read the answers. Browser automation would break OpenAI's terms and break whenever
the page changes. A ChatGPT subscription doesn't pay for API use either.

**The bar Ryan set:** answers as good as the ChatGPT app's search, not a quick search-snippet
summary. That means:
- searching several times for one question when needed, reading the pages, and cross-checking
  sources, the way ChatGPT search does
- giving a short spoken answer, with the sources kept in the log (not read aloud)
- deciding by itself when a question needs the web, without the user saying "search"

**Likely design (to confirm when we start):**
1. Add a `web_search` function tool to the Realtime session, with an env switch to turn it off like
   every other feature.
2. When the tool is called, `echo.py` asks OpenAI's Responses API, with its built-in `web_search`
   tool, on a strong model. That is the same kind of search the ChatGPT app uses, and it does the
   multi-step searching and reading.
3. Hand back a short answer plus sources. The Realtime model speaks it.
4. While it searches, say a quick "Let me look that up" so the silence doesn't read as a failure.

**Before building, check:** current Responses API web-search models and prices; how long a search
takes (spoken answers need to be quick); a side-by-side test of about 10 real questions against
the ChatGPT app, so we can judge "as effective" by the answers and not by guessing.

**Cheaper option, probably not good enough:** a raw search API (Serper), with the voice model
summing up the snippets itself. Rougher answers.

## 2. Say so when the API account is out of credit (proposed 2026-11-14)

On 2026-11-14 the OpenAI account ran out of credit at about 06:01
(`insufficient_quota.credit_balance_exhausted`). Nexus kept reconnecting every 15 s all day and
gave no sign of the problem, so it looked broken. Proposed fix: on a quota error, speak one short
line, send a push alert, and back off to retrying every few minutes. Needs an env switch. Not
approved yet.
