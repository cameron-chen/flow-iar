# Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning

Authors: Changyu Chen, Ramesha Karunasena, Thanh Hong Nguyen, Arunesh Sinha, Pradeep Varakantham

Many problems in Reinforcement Learning (RL) seek an optimal policy with large discrete multidimensional yet unordered action spaces; these include problems in randomized allocation of resources such as placements of multiple security resources and emergency response units, etc. A challenge in this setting is that the underlying action space is categorical (discrete and unordered) and large, for which existing RL methods do not perform well. Moreover, these problems require validity of the realized action (allocation); this validity constraint is often difficult to express compactly in a closed mathematical form. The allocation nature of the problem also prefers stochastic optimal policies, if one exists.

In this work, we address these challenges by (1) applying a (state) conditional normalizing flow to compactly represent the stochastic policy — the compactness arises due to the network only producing one sampled action and the corresponding log probability of the action, which is then used by an actor-critic method; and (2) employing an invalid action rejection method (via a valid action oracle) to update the base policy. The action rejection is enabled by a modified policy gradient that we derive. Finally, we conduct extensive experiments to show the scalability of our approach compared to prior methods and the ability to enforce arbitrary state-conditional constraints on the support of the distribution of actions in any state.

## Requirements
To install the necessary dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

While most dependencies can be installed using `pip`, some require alternative installation methods. For instance, to install `survae`, please refer to the instructions provided in its official repository (https://github.com/didriknielsen/survae_flows). We have included separate instructions for these dependencies in the `requirements.txt` file.

## Training & Evaluation
To train our model on various environments as reported in the paper, execute the script with the following command:
```bash
bash scripts.sh
```

You can also optionally run the command for a specific environment within the script.

For model evaluation, you can set the evaluation interval using the `--policy_eval_interval` flag. By default, we have set `--policy_eval_interval` to 5000, which means the model will be evaluated every 5,000 timesteps.

## Citation
If you find our work useful please cite:
```
@inproceedings{
  chen2023generative,
  title={Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning},
  author={Changyu Chen and Ramesha Karunasena and Thanh Hong Nguyen and Arunesh Sinha and Pradeep Varakantham},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## More information

Paper Link: https://arxiv.org/abs/2311.15341

Contact: cychen.2020@phdcs.smu.edu.sg