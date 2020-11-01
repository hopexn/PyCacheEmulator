from setuptools import setup

setup(name='CacheEmulator',
      version='1.0',
      description='Emulator for cache.',
      author='Haopeng Yan',
      author_email='yhp9523@qq.com',
      packages=['cache_emu', 'cache_baselines'],
      requires=['gym', 'numpy', 'torch'],
      )
