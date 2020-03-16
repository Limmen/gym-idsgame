from setuptools import setup

setup(name='gym_idsgame',
      version='1.0.0',
      install_requires=['gym', 'pyglet', 'numpy'],
      author='Kim Hammar',
      author_email='hammar.kim@gmail.com',
      description='IDS Markov Game RL Environment',
      license='MIT License',
      keywords='Cyber Security, Intrusion Detection, Markov Games, Reinforcement Learning, Q-learning',
      url='https://github.com/Limmen/gym-idsgame',
      download_url='https://github.com/Limmen/gym-idsgame/archive/1.0.0.tar.gz',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
  ]
)
