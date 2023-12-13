from json import loads

env_id_dict = {
    'Pistonball-v1': 'Pistonball-v0', 
    'Pistonball-v2': 'Pistonball-v1',
    'Pistonball-v3': 'PistonballNum10-v1',
    'PistonballCstr-v1': 'PistonballCstr-v0',
    'PistonballCstr-v2': 'PistonballCstr-v1',
    'SeqSSG-v1': 'SeqSSG-payoff-v1',
    'SeqSSG-v2': 'SeqSSG-payoff-v2',
    'SeqSSG-v3': 'SeqSSG-payoff-v3',
    'SeqSSG-v4': 'SeqSSG-payoff-v4',
    'SeqSSG-v5': 'SeqSSG-payoff-v5',
}

def add_env_args(parser):
    # Env params
    parser.add_argument('--env_id', type=str, default='LunarLander-v2')
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--n_eval_envs', type=int, default=1)
    parser.add_argument('--max_episode_steps', type=int, default=None)
    parser.add_argument('--vec_env_type', type=str, default='dummy', choices=['dummy', 'subproc'])
    parser.add_argument('--norm_obs_keys', nargs='*', type=str, default=None,
                        help='Which keys from observation dict to normalize. ' +
                            'If not specified, all keys will be normalized.')
    tmp_args, _ = parser.parse_known_args()

    if "ERSEnv" in tmp_args.env_id:
        import gym_ERSLE
        parser.add_argument('--convert_act', type=eval, default=False,)
    elif "SeqSSG" in tmp_args.env_id:
        from . import gym_seqssg
        parser.add_argument('--conf', type=str, default='automatic', 
                            choices=['automatic', 'manual', 'a', 'm'])
        parser.add_argument('--num_target', type=int, default=10)
        parser.add_argument('--graph_type', type=str, default='random_scale_free')
        parser.add_argument('--num_res', type=int, default=5)
        parser.add_argument('--groups', type=int, default=3, help='number of groups')
        parser.add_argument('--payoff_matrix', type=loads, default=None)
        parser.add_argument('--adj_matrix', type=loads, default=None)
        parser.add_argument('--norm_adj_matrix', type=loads, default=None)
        parser.add_argument('--def_constraints', type=loads, default=None)
        parser.add_argument('--no_constraint', type=eval, default=True)
        parser.add_argument('--init_state', type=str, default=None, 
                            help='initial state of the game. Format: ' + 
                                '"2,1,0" representing resources 0, 1, 2 are assigned to ' +
                                'target 2, 1, 0 respectively')
    elif "Pursuit" in tmp_args.env_id:
        from . import pursuit
    elif "Pistonball" in tmp_args.env_id: 
        from . import pistonball

def get_env_id(args):
    return env_id_dict.get(args.env_id, args.env_id)
