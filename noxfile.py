"""All the process that can be run using nox.

The nox run are build in isolated environment that will be stored in .nox. to force the venv update, remove the .nox/xxx folder.
"""

import nox

@nox.session(reuse_venv=True)
def docs(session: nox.session):
    """Build the documentation."""
    build = session.posargs.pop() if session.posargs else "html"
    session.install(".[docs]")
    session.run("sphinx-build", "-v", "-b", build, "docs", f"docs/_build/{build}")
