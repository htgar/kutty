# kutty

kutty is a local-first agent for small language models. It runs the model on your machine and gives it a shell, plain-text files, and one task at a time. There is no server, config file, or API key to manage. It builds to a single binary, works offline after the first model download, and keeps transcripts and memory in files you can read.

The design is deliberately small. Tool calls are structured, so the loop does not have to guess what the model meant. The model gets a shell and plain-text files because those are what it handles best. Runs are one-shot, with no daemon, no hidden state, and nothing to configure. The harness supplies the loop and tools; you supply the prompt and workflow.

This CLI is a starting point. The longer-term goal is a small core that can run anywhere and be embedded in other tools.

> **Experimental** There is a fair bit more functionality I would like to implement, so expect frequent breaking changes to the code. 

## Installation

Build from source:

```sh
cargo build --release
```

You need Rust 1.85 or newer, a C++ toolchain, CMake, and about 1.5 GB for the default model. On some systems (e.g. Fedora Atomic), bindgen may need the GCC include path:

```sh
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-redhat-linux/16/include"
cargo build --release
```

The first run downloads `openbmb/MiniCPM5-2B-GGUF` (`Q4_K_M`, ~1.5 GB) into the Hugging Face cache. After that it works offline.

## How to run

```sh
./target/release/kutty what is 2+2
./target/release/kutty list the files here using the shell tool
./target/release/kutty --model ./models/other.gguf what is in this directory
```

`--model` takes a local GGUF; otherwise the default model is downloaded. `--ctx` sets context size (minimum 512; default is the model's trained context, clamped to 512–4096). `--think` asks for a `thought` before the tool call. `--help` and `--version` do the usual.

Task words don't need quotes; they are joined with spaces. Quote a task containing shell syntax such as `?`, `*`, or `|`, and use `--` if it starts with a dash.

kutty runs in the current directory and is one-shot: it stops on `finish`, after 20 tool calls, on a repeated identical call, or before context overflow. Exit codes are 0 for success, 1 for runtime errors, and 2 for usage errors. Progress lines go to stdout as tools run and the answer is the final line.

Sessions and tool output are written under `.agents/`. The shell tool is not sandboxed; it runs with your permissions. File reads and writes are lexically confined to the working directory. Run it on data you can restore.

## Contributing

AI-assisted contributions are welcome. However you write the code, please make sure you've read, tested, and understood it before opening a PR. If you're unsure about a change, feel free to open an issue first. We can work out the approach together.

## Roadmap

### Active

- **Memory curation.** Curate memory from sessions so the agent improves across runs. Currently working on a more deterministic approach so smaller models can curate accurately.
- **Packaging and installation.** Provide packaged releases so installing kutty does not require a source build.

### Planned

- **Library core.** Move the engine, loop, and tools out of the CLI into a reusable crate that can run anywhere, once the CLI agent is stable.
- **Inference backend improvements.** Keep improving llama.cpp so it works across more models; a Rust-based backend is blocked until the Rust inference ecosystem matures.
- **Shell tool.** A sandboxed, portable shell to replace the current `sh -c` tool; currently testing different sandboxing options.

## License

MIT. See [LICENSE](LICENSE).
