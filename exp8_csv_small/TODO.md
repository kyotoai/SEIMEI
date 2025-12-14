### ISSUES #####
ALWAYS RUN : source /Users/juan/miniconda3/bin/activate
1. Knowledge should enhance score:
1.1 Prompt engineering (generate_knowledge_candidate)
1.2 Modify check_knowledge
1.3 prompt and improve it 


2. 
Make exp7/train_v3_eval.py (generate knowledge in all steps and see how much it improves)

Deeply understand exp7/train_v3.py and implement the following features in train_v3_eval.py
1. in this file, instead of running select_drifted_step to generate step, generate knowledge for first n_knowledge_steps steps and see how much score is improvevd by the knowledge compared to the base result.
2. when generating knowledge in each step, generate specified number of knowledge texts and get the best one to continue next step.
3. In the next step, use the agent outputs already generated in check_knowledge inference in the step before.
4. save the score results of all the inference made.

