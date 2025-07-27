import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyNitePrenguin",  # Changed from "PyNiteFEA"
    version="1.2.1",  # Custom version based on Pynite 1.2.0
    author="D. Craig Brinck, PE, SE (Original), Your Name (Modifications)",
    author_email="your.email@example.com",
    description="PyNitePrenguin: Enhanced 3D structural finite element library (based on Pynite)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Constratum/PyNitePrenguin",
    packages=setuptools.find_packages(include=['PyNitePrenguin', 'PyNitePrenguin.*']),
    package_data={
        'PyNitePrenguin': ['*html', '*.css', '*Full Logo No Buffer - Transparent.png']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Civil Engineering",
    ],
    install_requires=[
        'numpy>=1.19.0',
        'PrettyTable',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'all': ['IPython', 'vtk', 'pyvista[all,trame]', 'trame_jupyter_extension', 'ipywidgets', 'pdfkit', 'Jinja2'],
        'vtk': ['IPython', 'vtk'],
        'pyvista': ['pyvista[all,trame]', 'trame_jupyter_extension', 'ipywidgets'],
        'reporting': ['pdfkit', 'Jinja2'],
        'derivations': ['jupyterlab', 'sympy']
    },
    include_package_data=True,
    python_requires='>=3.7',
)
