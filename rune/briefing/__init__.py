"""The morning briefing: today's calendar, unread mail and the host's headlines, in one message.

Code reads every fact from the APIs (times, attendees, senders, conflicts). The model only orders
the mail and writes a one-line gist of each, as JSON, with no tools; what it writes is filtered
again before it is shown. Mail can carry instructions meant for an assistant, so nothing the model
reads can make it act, and a link or number it repeats from a mail never reaches the owner.
"""
