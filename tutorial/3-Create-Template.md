# Create Template

First, install the cookiecutter package using pip via

```shell
pip install cookiecutter
```

Once you've confirmed your installation, you can create the project by running 
```shell
cookiecutter https://github.com/AutoResearch/autora-template-cookiecutter
```

You will be prompted with a series of questions to customize your project. If a prompt displays, `N/A - Press Enter to Skip`, you can skip the question.

- For question **3**: *contribution_name*, provide only the contribution name without the prefix `autora-theorist`.
- For question **6**: *contribution_type*, choose 1 (theorist)
- For question **9**: *advanced_contribution_utilities*, choose 2 (advanced contribution). This will setup GitHub actions and pre-commit hooks
- For question **11**: *use_current_directory*, choose 2 (yes). This will setup the project in the current folder. Make sure that the contribution name and the name of the repository are the same.

After running the command, you will see several new files and directories in the folder. Let's go through them step-by-step to understand their purpose and structure:

***
Next: [4-Project-Structure](./4-Project-Structure.md)
***
**Navigate Through the Tutorial**:
- Use the navigation buttons in the preview pane to move through the tutorial steps.
- After pressing on links to navigate, click on the left editor window to refocus.