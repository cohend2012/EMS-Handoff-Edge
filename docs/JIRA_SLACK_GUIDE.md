# Team guide: the project's Slack and Jira

Written 2026-10-29. The project coordinates in two tools that Daniel administers:

| tool | address | used for |
|---|---|---|
| Slack | axonemshandoff.slack.com | day-to-day: the hourly progress note, threads on results, files and links |
| Jira | axonemshandoff.atlassian.net | the work list: tickets with owner, status and due date |

## 1. Getting in

Access is by invitation to one specific email address. The invite goes to the address the person asks for,
and that is the only address the workspace recognizes afterwards.

**Slack.** Open the invite email, click *Join now*, and create the account with the invited address (do not
sign in with an older Slack account). Password or emailed code, either works. On the phone app choose
*Sign in with email* with the same address and the workspace appears in the list. "You don't have an
account on this workspace" means a different address was used: sign out and use the invited one.

**Jira.** Open the invite email, click *Accept invitation*, create or sign in to an Atlassian account with
the invited address, and bookmark the site. The Jira Cloud phone app signs in the same way.

**For the administrator:** Slack, workspace menu, *Invite people*, enter the address. Jira, *Settings*,
*User management*, *Invite users*, enter the address. If a person was migrated to a new email, send a
fresh invite to the new address; the old invite keeps failing.

## 2. How we use Slack

- One project channel carries the hourly progress note from the build: what finished with its numbers,
  what is running, what is blocked, what is next. The same text goes by email to Daniel.
- Replies go in the thread under the post they are about, so a question stays next to its result.
- Files and links to reports go in the channel, not in email.
- An @mention when someone's eyes are needed; direct messages for anything private.

## 3. How we use Jira

- Every task is a ticket with an owner, a status (To do, In progress, Done) and a due date; the board shows
  what is moving.
- Plans become tickets: one per lever in `docs/plans/HORSEPOWER_GOALS.md` and one per radio experiment in
  `docs/plans/RADIO_AI_OFFGRID.md`; the statistics work for the study (intervals on every reported
  difference, paired comparisons, annotator agreement on the references) goes to the biostatistician.
- Questions and results go in the ticket's comments so the discussion stays with the work; a ticket moves
  to Done when its result is in a report with a commit hash.

## 4. The rhythm

A morning glance at the Slack project channel; questions in the thread of the relevant post; tasks and
their status in Jira; email only for people outside the project.

## 5. The assistant and these tools

The build assistant (Claude Code on the Jetsons) can post the hourly note to Slack and keep Jira tickets
current once Daniel connects it to his own accounts on both. It acts there only when asked, apart from the
agreed hourly note, and it never shares patient audio or transcripts outside the project's private pages.
