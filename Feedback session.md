# Feedback session



1. Change the research question to something that is inextricably linked to the scope of interest. A first example came into my mind: *See how the inference speed and accuracy increase by lowing the number of FLOPs or model's capacity*
2. Make sure that all the proofs were "measured" by using only single metric \~ **THEIRS \~** and include these in tables, the first two bullet points as an aggregate and the third as a verification:

   1. Figure out how to calculate the number of FLOP(s) again;
   2. Figure out how to determine the DICE score or make sure that your calculations are sound;
   3. Take into consideration the model's size and include this in the table as well
3. Make sure that all relevant findings (e.g., illustrations, figures, tables), are included in the paper itself (e.g., every chapter setting aside the APPENDIX)

   1. Table related issues:

      1. Make sure the table is included in the paper itself;
      2. They don't seem to like the tagging/labeling for each model so instead they recommended inserting new table columns each describing model attributes (e.g., image size, optimizer);
      3. The table appearance is altered by its uncommonly lengthy size so try to work on that;
      4. Make sure the measurements units for energy are provided;
      5. Make the table visually appealing;

4\. Make sure that you describe the evolution of your model entirely and precisely, beginning with a clear baseline and finishing off with a nicely outlined prototype; Showing the model improvement must be geared to one single metric (DiceLoss/FLOPs)



5\. Create no-brainers executables (e.g., executable predictions based on sample hosted somewhere and preferably everything runnable in CLI) and crystal clear(actionable and followable README.md);





**NEXT STEPS starts;**

6\. What sort of information is already available over there? ; Check with Tim if the new research question is something that is in match with their scope of interest; 

**NEXT STEPS ends;**





7\. Stick to this table of contents: 

a) Three pages for experiments, conclusions and discussions;

b) One page for introduction;

c) Rest is appendix; 

d) Refine the title?

