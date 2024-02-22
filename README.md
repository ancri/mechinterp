# AISC Out-Of-Context Interpretability
[Feb 17, 2024] description of experiment in asic_toymodel/src/oocl.py  
- the idea is to create a setup where the task is much more easily completed if a connection is made between two (mutually out of context) sections of data
- the task performed is modular arithmetic, specifically `(x^2 + y^2) mod p`, where p is a prime
- there are a total of 2 * p + 4 tokens in the vocabulary
- tokens 0,...,p-1 (operands group 1) are the x's and y's (and results) used in the binary operation table
- tokens p,...,2*p-1 (operands group 2) could be consider elementwise "aliases" of tokens 0,...,p-1 which naturally also create an equivalent table
- the training dataset consists of:
  - a small number of elements from the binary operation table using operands group 1 (the majority are held out for the test set)
  - a small number of elements from the operands group 2 operation table (the majority are held out for the test set)
  - all the linkages between element i (from group 1) and i+p (from group 2)
- the validation / test dataset consists of:
  - the held out (majority) elements from binary op table 1
  - the held out (majority) elements from binary op table 2
- the elements held out from the two binary op tables are mutually indepenedent, such that if the linkages are understood and properly interpreted the model should have a much easier time training (as if less of the binary op table was held out)
- the training loss is the sum of 3 losses: `k1 * loss_op1 + k2 * loss_op2 + k3 * loss_linkages`, where k1, k2, and k3 are coefficients, loss_op1 is the loss on the first binary op table and loss_op2 is the loss on the second binary table, and loss_linkages is the loss on the linkages
- by tuning knobs k1, k2 and k3 we can allow the model to learn via the linkages
- what we would like ideally is to find a configuration where we hold out a certain fraction of data such that with (k1, k2, k3) set to (1, 0, 0) the model can't train, but with (1, 1, 1) the model does train
- the naming of models in wandb is: oocl_ssq_p_holdout1_holdout2_k1_k2_k3 where holdout1 is the fraction held out from binary op table 1, holdout2 is the fraction held out from binary op table 2, and the rest have already been explained
