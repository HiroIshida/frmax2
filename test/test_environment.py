import numpy as np

from frmax2.environment import GaussianEnvironment


def test_gaussain_environment():
    # test volume
    env_1d = GaussianEnvironment(10, 1)
    np.testing.assert_almost_equal(env_1d.evaluate_size(np.zeros(10)), 2 * 1.0)

    env_2d = GaussianEnvironment(10, 2)
    np.testing.assert_almost_equal(env_2d.evaluate_size(np.zeros(10)), np.pi * 1.0**2)

    env_3d = GaussianEnvironment(10, 3)
    np.testing.assert_almost_equal(env_3d.evaluate_size(np.zeros(10)), 4 / 3 * np.pi * 1.0**3)

    env_2d_hollow = GaussianEnvironment(10, 2, with_hollow=True)
    np.testing.assert_almost_equal(
        env_2d_hollow.evaluate_size(np.zeros(10)), np.pi * (1.0**2 - 0.2**2)
    )

    # test inside
    for dim in range(1, 20):
        env = GaussianEnvironment(10, dim)
        env_hollow = GaussianEnvironment(10, dim, with_hollow=True)

        e = np.zeros(dim)
        e[0] = 1.0 - 1e-2
        x = np.hstack([np.zeros(10), e])
        assert env.isInside(x)
        assert env_hollow.isInside(x)

        e[0] = 1.0 + 1e-2
        x = np.hstack([np.zeros(10), e])
        assert not env.isInside(x)
        assert not env_hollow.isInside(x)


def test_environment_with_ignored_axes():
    consider_axes = [0, 2]
    env_3d = GaussianEnvironment(10, 5, error_consider_axes=consider_axes)
    size = env_3d.evaluate_size(np.zeros(10))
    np.testing.assert_almost_equal(size, np.pi * 1.0**2 * 3.0**3)

    for _ in range(100):
        param = np.zeros(10)
        err = np.random.randn(5)
        err[np.array(consider_axes)] = 0.0
        assert env_3d.isInside(np.hstack([param, err]))


if __name__ == "__main__":
    test_gaussain_environment()
    test_environment_with_ignored_axes()
