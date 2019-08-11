from distutils.core import setup

def readme():
    """
    Function to read the long description for the MLROSe package.
    """
    with open('README.md') as _file:
        return _file.read()



setup(
  name = 'mlswarm',         # How you named your package folder (MyLib)
  packages = ['mlswarm'],   # Chose the same as "name"
  version = '0.21',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'This package trains neural networks using swarm-like optimization algorithms',   # Give a short description about your library
  long_description=readme(),
  long_description_content_type='text/markdown',
  author = 'Rafael Cabral',                   # Type in your name
  author_email = 'rafael.medeiroscabral@kaust.edu.sa',      # Type in your E-Mail
  url = 'https://github.com/rafaelcabral96/mlswarm',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/rafaelcabral96/mlswarm/archive/v021.tar.gz',    # I explain this later on
  keywords = ['machine learning', 'neural networks',   'optimization', 'particle swarm',  'derivative free'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'sklearn',
          'pandas',
          'matplotlib'
      ],
  python_requires='>=3',
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
