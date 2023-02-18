from setuptools import setup

setup(
    name='jax-rwkv',
    version='0.0.1',    
    description="This is a jax implementation of BlinkDL's RWKV.",
    url='https://github.com/tensorpro/jax-rwkv',
    author='tensorpro',
    author_email='tensorpro@gmail.com',
    license='Apache License 2.0',
    packages=["prwkv"],
    install_requires=['tokenizers',
                      'jax',
                      'optax',
                      'flax'               
                      ],
    include_package_data = True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'License :: OSI Approved :: Apache Software License',
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.9'
    ],
)