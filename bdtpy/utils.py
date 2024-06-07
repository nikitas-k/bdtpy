from IPython.display import display, Markdown

def display_eq(cls):
    """Print the docstring of a class."""
    docstring = cls.__doc__
    if docstring:
        display(Markdown(f"## `{cls.__name__}`\n\n{docstring}"))
    else:
        print(f"No docstring found for class {cls.__name__}.")