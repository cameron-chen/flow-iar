from policies.act_mask.base import ActorCriticActMaskPolicy
from policies.flow_policy.base import ActorCriticFlowPolicy
from policies.joint_mlp.base import ActorCriticJointPolicy


def add_policy_args(parser):
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--log_weight_grad', type=eval, default=False)
    parser.add_argument('--log_model_structure', type=eval, default=False)
    parser.add_argument('--model_summary', type=eval, default=False)
    parser.add_argument('-fea_ext','--features_extractor_class', type=str, default=None)
    parser.add_argument('--embedding_vars_gnn_extractor', nargs='*', type=int, default=None,
                        help='(List[Tuple]) Number of classes and  dimension of embedded variables in GNN extractor')
    parser.add_argument('--apply_bn_gnn_extractor', type=eval, default=True,
                        help='Whether to apply batch normalization to the GNN.')
    tmp_args, _ = parser.parse_known_args()
    # Assumes our own policies: flow, rnn, alpha_projection and the policies
    # in SB3
    if tmp_args.policy == 'flow':
        parser.add_argument('--flow_type', type=str, default=None, choices=['ar', 'shallow_coupling','coupling', 'res'])
        parser.add_argument('--flow_net_hidden_size', type=int, default=128)
        parser.add_argument('--context_size', type=int, default=64,
            help="size of the condition for a conditional flow")
        parser.add_argument('--num_flow_layers', type=int, default=4)
        parser.add_argument('--batch_size_flow_updating', type=int, default=64)
        parser.add_argument('--n_iters_flow_pretraining', type=int, default=0)
        parser.add_argument('--val_steps', type=int, default=1)
        parser.add_argument('--val_batch_size', type=int, default=64)
        parser.add_argument('--elbo_steps', type=int, default=1)
        parser.add_argument('--n_samples_prob_est', type=int, default=4)
        parser.add_argument('--flow_base_dist', type=str, default='cond_mean_std_gauss',
                            choices=['cond_mean_std_gauss', 'cond_mean_gauss','cond_gauss'])
        parser.add_argument('--act_encoding_scheme', type=str, default='one_hot',
                            choices=['one_hot', 'cartesian_product'])
        parser.add_argument('--pol_grad_G', type=eval, default=False)
        parser.add_argument('--elbo_Q', type=eval, default=False)
        parser.add_argument('--posterior_type', type=str, default="normal")
        parser.add_argument('--num_posterior_layers', type=int, default=2)
        parser.add_argument('--log_flow_dist', type=eval, default=False)
        parser.add_argument('--has_act_corr', type=eval, default=False)
        parser.add_argument('--act_corr_prot', type=str, default=None,
                            choices=['flow', 'val', 'flow_joint', 'val_joint'])
        parser.add_argument('--lmd_corr', type=float, default=1.0)
        parser.add_argument('--noise_std', type=float, default=0.0)
        parser.add_argument('--rep_reg', type=int, default=-1)
        parser.add_argument('--cond_bijection', type=eval, default=False)
        parser.add_argument('--sandwich_evidence', type=eval, default=False)
        parser.add_argument('--ensemble_mode', type=str, default=None)
    elif tmp_args.policy == 'rnn':
        parser.add_argument('--placeholder', type=int, default=1) 
    elif tmp_args.policy == 'alpha_projection':
        parser.add_argument('--placeholder', type=int, default=1)
    else:
        parser.add_argument('--placeholder', type=int, default=1)

def get_policy_id(args):
    if args.policy == 'flow':
        return 'flow'
    elif args.policy == 'mask':
        return 'mask' 
    elif args.policy == 'joint':
        return 'joint'
    elif args.policy == 'rnn':
        return 'rnn'
    elif args.policy == 'alpha_projection':
        return 'alpha'
    elif args.policy == 'MlpPolicy':
        return 'sb3_mlp'
    elif args.policy == 'CnnPolicy':
        return 'sb3_cnn'
    elif args.policy == 'MultiInputPolicy':
        return 'sb3_multi_input'
    else:
        raise ValueError('No policy named f{args.policy}')


def get_policy(args):
    if args.policy == 'flow':
        return ActorCriticFlowPolicy
    elif args.policy == 'mask':
        return ActorCriticActMaskPolicy
    elif args.policy == 'joint':
        return ActorCriticJointPolicy
    elif args.policy == 'rnn':
        return 'rnn_class'
    elif args.policy == 'alpha_projection':
        return 'alpha_class'
    elif args.policy in ['MlpPolicy', 'CnnPolicy', 'MultiInputPolicy']:
        return args.policy
    else:
        raise ValueError('No policy named f{args.policy}')
