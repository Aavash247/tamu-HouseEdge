# House Edge guide (part 3)

| Probabilities don't add up to one | Check column order, normalize each row, and avoid rounding too much. Find the cause of any NaNs. |
| Extra columns | Remove dataframe indices. With pandas, use `index=False`. |
| Output folder already exists | Pick a new `--out` folder. |
| Points differ from the `score` field | Use `technical_points` for the round score. The older diagnostic has a different scale. |
| Can I use coding assistants or automated tuning? | Yes. You're responsible for the results and for following the data rules. |
| Do I need RL, a neural network, or a GPU? | No. Any model that produces the required probabilities can compete. |
| Can I use outside data? | Use organizer-released data. Private test answers and organizer files are off limits. |
| Can I calculate my final score locally? | You can score validation predictions. Test answers stay private. |
| Does the slot-machine demo count? | No. It uses practice odds; submit your CSV to compete. |

### Uploads and code checks

| Problem | What to check |
| --- | --- |
| Form asks me to sign in | File uploads require a Google account. If your account is blocked, contact organizers before cutoff. |
| No confirmation | Attaching files is not enough: click Submit. If unsure, have organizers check the response log. |
| Wrong query IDs | Main, adaptation, validation, and test have different IDs. Use the correct kit and split. |
| Upload too large | CSV: 10 MB; ZIP: 100 MB. Remove unrelated files, without removing required model files. |
| ZIP fails validation | Check the root files and paths. Use `build_entry.py` to package it. |
| Module not found during reproduction | Include your source modules and list third-party packages with exact versions in `requirements.txt`. |
| Can't write a file | Use `/tmp` for scratch and `/output` for results. `/submission` and `/data` are read-only. |
| Recreated predictions differ | Check the checkpoint, preprocessing, column order, precision, and seeds against your submitted CSV. |
| Timeout | Load a saved checkpoint and stay within the runtime budget. |
| No score or code-check status | Expected. Forms confirms receipt; organizers validate and rerun code afterward. |
| A replacement is invalid | That latest response still counts. Fix and resubmit before cutoff; older entries aren't automatically restored. |
| Deadline passed or Form closed early | Contact organizers through the announced help route. Don't assume late files will count. |
| Second kit unavailable | It releases at 9 PM Saturday. Main code checks do not control the release. |

When asking for help, include the command, Python version, round, and exact error. Keep your account details private. Share code or model files only if you intend others to see them.
