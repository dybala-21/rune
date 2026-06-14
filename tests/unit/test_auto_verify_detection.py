

def test_passed_test_count_parsing():
    from rune.agent.auto_verify import passed_test_count
    assert passed_test_count("3 passed in 0.01s") == 3
    assert passed_test_count("1 passed, 2 warnings in 0.1s") == 1
    assert passed_test_count("47 passed in 2.3s") == 47
    assert passed_test_count("no tests ran in 0.0s") is None
    assert passed_test_count("") is None
