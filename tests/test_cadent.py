import numpy as np
import pytest

from cadent.agents import CADENT, METHODS, QLearningConfig
from cadent.envs import ENVIRONMENTS, make
from cadent.teacher import TeacherConfig, TeacherKnowledge, distill, softmax, teacher_knowledge

RNG = np.random.default_rng(0)


def _knowledge(env_name):
    return teacher_knowledge(make(env_name, "teacher"), TeacherConfig(episodes=60), np.random.default_rng(1))


@pytest.mark.parametrize("name", ENVIRONMENTS)
def test_outcome_probabilities_sum_to_one(name):
    env = make(name)
    s = env.reset()
    for a in range(env.n_actions):
        assert sum(p for _, p in env.outcomes(s, a)) == pytest.approx(1.0)


def test_factory_converts_carried_wood():
    env = make("blind_craftsman", "teacher")
    s = ((3, 3), (2, 0), 3)                     # two units of wood, next to the factory at (3, 4)
    (s2, r), = [(o, env.reward(s, o)) for o, _ in env.outcomes(s, env.actions.index("right"))]
    assert s2[1] == (0, 2) and s2[-1] == 5 and r == 1.0


@pytest.mark.parametrize("name", ENVIRONMENTS)
def test_distilled_transition_values_exist(name):
    """Eq. 4: the teacher's knowledge must contain values for real automaton transitions."""
    k = _knowledge(name)
    assert any(q != q2 for q, q2 in k.q_ad)
    for pi in k.policy.values():
        assert pi.sum() == pytest.approx(1.0)


def test_trust_starts_high_and_drops_with_volatility():
    env = make("blind_craftsman")
    k = TeacherKnowledge({}, {}, 1.0, {})
    agent = CADENT(env, QLearningConfig(), RNG, k, eta=0.5, k=5.0, theta=0.5)
    s = env.reset()
    assert agent.trust(s, 0, 0.0) > 0.9
    for _ in range(10):
        omega = agent.trust(s, 0, 10.0)
    assert omega < 0.01


def test_fixed_trust_ablation():
    env = make("blind_craftsman")
    agent = CADENT(env, QLearningConfig(), RNG, TeacherKnowledge({}, {}, 1.0, {}), trust="fixed", fixed_trust=0.5)
    assert agent.trust(env.reset(), 0, 100.0) == 0.5


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_trains(method):
    env = make("blind_craftsman", "teacher")
    k = _knowledge("blind_craftsman")
    cls = METHODS[method]
    agent = cls(env, QLearningConfig(), np.random.default_rng(2)) if method == "no_transfer" \
        else cls(env, QLearningConfig(), np.random.default_rng(2), k)
    log = agent.train(30)
    assert len(log.rewards) == 30 and all(np.isfinite(log.rewards))


def test_distill_averages_the_triggering_state_actions():
    env = make("blind_craftsman", "teacher")
    s = ((1, 0), (0, 0), 0)                      # wood at (1, 1): "right" triggers 0 -> 1
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_ad, policy = distill(env, {s: q})
    assert q_ad[(0, 1)] == pytest.approx(4.0)
    assert policy[0] == pytest.approx(softmax(q))
