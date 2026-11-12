def test_package_imports():
    import jobagent              # try to open the drawer
    assert jobagent.__doc__      # check its label (the docstring) isn't empty