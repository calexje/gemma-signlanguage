# GEMA4

Callum has developed a ASL transcriber that uses gemma4 to complete a correct
transcription. The Gesture Education for Mentoring ASL 4, the GEMA4.

# What is GEMA4?

How the program works is very simple:

- Input is the webcam or any connected computer camera, the camera must point
at the ASL speaker
- To begin transcribing, the speaker must hold up their palm and the a buffer
of the letters transcribed is created.
- Video feed goes through an algorithm to recognise the ASL letters and write
them to a buffer
- When the ASL speaker decides to end the phrase, gemma 4 will check
spellings and summarize the conversation.

This program can help when people that can only speak ASL (i.e. deaf people) to
be able to go back on a conversation that just happened, as if you miss
anything said by the speaker you cannot go back on that conversation
normally. With GEMA4, a user can go back at any time to read or
listen to the conversation.

A native ASL speaker can also use this to communicate with non-ASL speakers.
For example, a deaf ASL speaker can speak into a small device like their phone
or their laptop and the non-deaf non-ASL person can read the transcription
enabling more accessibility to deaf users.

# How does it work

The program is written in python and uses opencv to ingest the video feed. Look
at the graph below get a better understanding on how it works.

![A digram showing how GEMA4 works. It has an entry and exit note and shows
the process taking in a video feed, using opencv to track hand signs,
transcribing using ASL dictionary and fixing typos and summarizing using
Google's gemma4](./GEMA4_Diagram.png)

# Expanding on this

We have a couple of ideas on how to expand this to have even more
accessibility:

- For blind or reduced eyesight people on the listening end, the program could
use a text-to-speech model in other to more naturally converse between deaf ASL
speakers and blind non-ASL speakers.
- A small neck-worn device with a fisheye lens or glasses like the meta rayban
glasses could read ASL from a speaker and automatically transcribe either onto
a speaker or a screen.