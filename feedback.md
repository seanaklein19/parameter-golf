# Operator Feedback

Write notes here anytime. The agent reads this file every iteration.
Delete or clear entries once they've been addressed.

Think about increasing iterations or max_wallclock_seconds if you think you can push BPB further with more training at some point
                                                            
  ## Guidance                                                                                                                            
  - Vary the iterations config between runs. Try short runs (200-500 steps) for quick directional signal, longer runs (1000-2500) when
  you need to confirm a finding.
  - Push against the contest constraints: 16MB artifact budget and 10 min on 8xH100. Explore larger models that get close to the limit.
  - Prioritize breadth early: try many different ideas with short runs before committing to long convergence runs.                       
  - Don't just tune one knob at a time — test bold architectural changes (different dims, layer counts, attention configs).
                                                                                                                             
