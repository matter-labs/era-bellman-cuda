# bellman-cuda

[![Logo](eraLogo.svg)](https://zksync.io/)

zkSync Era is a layer 2 rollup that uses zero-knowledge proofs to scale Ethereum without compromising on security or
decentralization. Since it's EVM compatible (Solidity/Vyper), 99% of Ethereum projects can redeploy without refactoring
or re-auditing a single line of code. zkSync Era also uses an LLVM-based compiler that will eventually let developers
write smart contracts in C++, Rust and other popular languages.

bellman-cuda is a library implementing GPU-accelerated cryptographic functionality for the zkSync prover. 

### Building the library
The library can be build by executing these steps:

#### Initialize git submodules
`git submodule update --init --recursive`
#### Generate the build configuration by executing
`cmake -B./build -DCMAKE_BUILD_TYPE=Release`
#### Build the binary by executing
`cmake --build ./build`

The library binary can be found in the `./build/src` folder. Change the path in the above commands if a different build location is desired.

By default, the library is build for Compute Architecture 8.0.
If a different Compute Architecture is required, the [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html) variables can be set to the desired architecture(s) during the build configuration generation step.

Example for Compute Architecture 8.6: 
```
cmake -B./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```

### Executing Tests
By default, the tests binary is not compiled.

This can be changed by setting the variable `BUILD_TESTS` to `ON` in the build configuration step like this:

`cmake -B./build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON`.

Then, after executing the build step, the tests binary is located in the `./build/tests` folder and can be executed by calling:

`./build/tests/tests`.  

## License

The zkSync Era prover is distributed under the terms of either

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Official Links

- [Website](https://zksync.io/)
- [GitHub](https://github.com/matter-labs)
- [Twitter](https://twitter.com/zksync)
- [Twitter for Devs](https://twitter.com/zkSyncDevs)
- [Discord](https://discord.gg/nMaPGrDDwk)
