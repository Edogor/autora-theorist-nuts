
# Project Structure

## Implement Your Code

You may implement your code in the `__init__.py` located in the respective feature folder in ``src/autora``.

Please refer to the following guides on implementing

* [theorists](theorist.md)
* [experimentalists](experimentalist.md)
* [experiment runners](experiment-runner.md)

## Tests

It is highly encouraged to add unit tests to ensure your code is working as intended. These can be [doctests](https://docs.python.org/3/library/doctest.html) or test cases in `tests/test_your_contribution_name.py`.

## Documentation

It is highly encouraged that you add documentation of your package in `docs/index.md`. You can also add new or delete unnecessary pages 
in the `docs` folder. However you structure your documentation, be sure that structure is reflected in the `mkdocs.yml` file.

You are also encouraged to describe basic usage of your module in the 
python notebook `Basic Usage.ipynb` in the `docs` folder. Finally you can outline the basic setup of your module in 
the `docs/quickstart.md` file.

## Dependencies

In the `pyproject.toml` file, add the new dependencies under `dependencies`. Here you can also add optional dependencies (for example dependencies that are only needed in development like mkdocs to build documentation)

Install the added dependencies
```shell
pip install -e .
```

## GitHub Workflows

In the `.github/workflows` folder, you find github actions.
`python-publish.yml` is an action that enables automatic publishing:

### Publishing Via GitHub Actions

1. Add an API token to the GitHub Secrets
    - Create a [PyPI account](https://pypi.org/) if you don't have one already.
    - Once you have an account, generate an API token for your account.
    - In your GitHub repository, go to `Settings`.
    - Under `Secrets and variables` in the left-hand menu, select `Actions`. 
    - Create a new secret named `PYPI_API_TOKEN` and paste in your PyPI API token as the value.

2. Create a new release
    - Follow the steps outlined in the [GitHub documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) for creating a new release. 
    - Once you create a new release, the GitHub Action will automatically trigger, and your package will be built and published to PyPI using the provided API token.


## Pre-Commit Hooks

`.pre-commit-config.yml` is a [`pre-commit`](https://pre-commit.com) file.

Pre-commit hooks are programs which run before each git commit, and can read and potentially modify the files which are to be committed. 

We use pre-commit hooks to:
- enforce coding guidelines, including the `python` style-guide [PEP8](https://peps.python.org/pep-0008/) (`black` and `flake8`),
- to check the order of `import` statements (`isort`),
- to check the types of `python` objects (`mypy`).

## Handling Pre-Commit Hook Errors

If your `git commit` fails because of the pre-commit hook, then you should:

1. Run the pre-commit hooks on the files which you have staged, by running the following command in your terminal: 
    ```zsh
    $ pre-commit run
    ```

2. Inspect the output. It might look like this:
   ```
   $ pre-commit run
   black....................Passed
   isort....................Passed
   flake8...................Passed
   mypy.....................Failed
   - hook id: mypy
   - exit code: 1
   
   example.py:33: error: Need type annotation for "data" (hint: "data: Dict[<type>, <type>] = ...")
   Found 1 errors in 1 files (checked 10 source files)
   ```
3. Fix any errors which are reported.
   **Important: Once you've changed the code, re-stage the files it to Git. 
   This might mean un-staging changes and then adding them again.**
4. If you have trouble:
   - Do a web-search to see if someone else had a similar error in the past.
   - Check that the tests you've written work correctly.
   - Check that there aren't any other obvious errors with the code.
   - If you've done all of that, and you still can't fix the problem, get help from someone else on the team.
5. Repeat 1-4 until all hooks return "passed", e.g.
   ```
   $ pre-commit run
   black....................Passed
   isort....................Passed
   flake8...................Passed
   mypy.....................Passed
   ```

****
