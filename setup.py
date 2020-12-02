from setuptools import setup

setup(name='PyCacheEmulator',
      version='1.2',
      description='Emulator for cache',
      author='Haopeng Yan',
      author_email='yhp9523@qq.com',
      packages=['py_cache_emu'],
      requires=['gym', 'numpy', 'torch', 'pyyaml', 'pandas'],
      )
