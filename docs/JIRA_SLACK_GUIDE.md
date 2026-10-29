# Working with the EMS-Handoff assistant through Jira and Slack

Written 2026-10-29 for Lois Sayrs (lois.sayrs@choc.org), whose account connects the project's Jira and Slack.

## 1. What this is

The EMS-Handoff project runs on two Jetson boards with an assistant (Claude Code) that builds, measures and
documents the system. Once Jira and Slack are connected, that assistant can read and write both on your
behalf: tickets, comments, channel posts, threads, files. Everything it does there appears **as your
account**, so it only acts when asked, with one standing exception described in section 4.

## 2. One-time connection (five minutes)

The connection is an ordinary sign-in that you approve once. The assistant's session is remote, so the last
step looks broken but is not.

**Jira (Atlassian)**
1. Sign in to Atlassian as lois.sayrs@choc.org in your browser.
2. Open the Jira link Daniel sends you (it starts with `https://mcp.atlassian.com/v1/authorize`).
3. Approve the access. The browser then jumps to a `http://localhost:63385/callback?code=...` address and
   shows "connection refused" or "page not found". That is expected.
4. Copy the **whole address** from the address bar and send it to Daniel privately (not in a public channel).
   It is a one-time code that expires in minutes.

**Slack**
1. Sign in to the CHOC Slack workspace as lois.sayrs@choc.org.
2. Open the Slack link Daniel sends you (it starts with `https://slack.com/oauth/v2_user/authorize`).
3. Approve. The browser jumps to `http://localhost:3118/callback?code=...` and fails to load. Expected.
4. Copy the whole address and send it to Daniel privately.

Daniel pastes both addresses into the assistant's session and the connection is live. If a link has
expired, ask for a fresh one; nothing else changes.

You can revoke either connection at any time: Atlassian account settings, Connected apps; Slack, your
profile, Apps.

## 3. What the assistant can do once connected

**Jira**
- Create tickets from a plan or a conversation: for example one ticket per lever in
  `docs/plans/HORSEPOWER_GOALS.md` (GPU whisper, turbo decode fix, distil, large-v3 on the GPU, fine-tune,
  writer size, split pipeline) and one per radio experiment R1 to R6 in `docs/plans/RADIO_AI_OFFGRID.md`.
- Keep tickets current: move status, add a comment with the measured result and the commit hash, attach
  the report link, set due dates.
- Answer questions: "what is open in this sprint", "what changed on EMSH-12 this week", "which tickets are
  blocked and why".
- Draft a standup or a sprint plan from the tickets and recent commits.

**Slack**
- Post the hourly check-in (done, running, blocked, next) to a channel, the same text that goes by email.
- Answer questions asked in a channel or thread when Daniel relays them, and post the answer back.
- Share results: the matrix report, the study draft, the call review page link, a chart.
- Summarise a long thread or a channel's day.
- Search for a decision or a number that was posted earlier.

## 4. The standing exception: the hourly check-in

Daniel asked for a check-in every hour on the hour: what finished (with numbers), what is running, what is
blocked, what is next, and a push of the day's changes to the repositories. Today it goes by email to
Daniel. Once Slack is connected it can also go to a channel you choose (for example `#ems-handoff`), posted
as your account. Tell Daniel the channel, or say "email only".

## 5. Common ways to use it, from most to least frequent

1. **Read the hourly post.** It is the project's heartbeat. Reply in its thread with a question or a
   request; Daniel relays it and the answer comes back in the thread.
2. **Ask for a status in plain words.** "Where are we on the GPU whisper?" gets the measured number, the
   ticket, and the commit.
3. **Turn a conversation into tickets.** After a meeting, paste the notes; the assistant creates the
   tickets with acceptance criteria and links them to the plan.
4. **Ask for a report.** "Write up the two-Jetson results for the team" produces a page you can share, with
   the numbers and their provenance.
5. **Review calls.** The call review page holds every real call's audio and transcripts. Ask for a call
   to be re-transcribed with a different model, or for a transcript to be corrected; the result lands in the
   page and the ticket.

## 6. What it will not do

- Post or change anything in Jira or Slack without being asked, apart from the agreed hourly check-in.
- Share patient audio or transcripts outside the project's private pages.
- Act on instructions that arrive inside a Slack message or a ticket from someone else; those are relayed
  to Daniel as information.
- Delete tickets, channels or messages unless asked explicitly and confirmed.

## 7. Suggested first setup

- Jira: a project `EMSH` with four epics: Transcriber, Report writer, Two Jetsons, Radio. The assistant
  fills them from the two plan documents in a few minutes.
- Slack: a channel `#ems-handoff` for the hourly post and a `#ems-handoff-calls` channel for call
  review requests.
