# .NET SDK compatibility

Version 0.33.1 includes three managed assemblies with the same public SDK:
.NET Standard 2.0, .NET Framework 4.8 and .NET 8.

| Application target | Selected TurboJev assembly |
| --- | --- |
| .NET Framework 4.8 / 4.8.1 | `net48` |
| .NET Core 3.1 | `netstandard2.0` |
| .NET 5, 6, 7 | `netstandard2.0` |
| .NET 8, 9, 10 | `net8.0` |

Compatibility with an older runtime does not change that runtime's support
lifecycle. Framework versions below 4.8 are not qualified by TurboJev.
Core 2.x and 3.0 cannot build with the SDK's current transitive dependencies.
Core 3.1 and .NET 5 are locally qualified legacy configurations; the
System.Text.Json 8 dependency emits a support warning for these runtimes.
The compatibility test leaves that warning visible.

## Install

For an SDK-style application, install the package and choose the process RID:

```powershell
dotnet add package TurboJev
dotnet run --runtime win-x64
```

For Framework 4.8, use Visual Studio's NuGet Package Manager or its console:

```powershell
Install-Package TurboJev
```

Use `PackageReference` to receive the native-copy build targets. For Framework
executables, enable `AutoGenerateBindingRedirects` and
`GenerateBindingRedirectsOutputType`. NuGet supplies the JSON and async-interface
dependencies. C# language syntax is separate from the target framework; using
this SDK does not require upgrading an application's code to C# 12.

## Load a model

This method can be used in a Framework, Core or modern .NET application. Await it
from a UI event handler to keep the UI responsive:

```csharp
using System;
using System.Threading.Tasks;
using TurboJev;

public static class TicketDemo
{
    public static async Task RunAsync()
    {
        using (var engine = await TurboJevEngine.LoadAsync(new
        {
            model = "model.gguf",
            device = "cpu",
            nThreads = 4
        }))
        {
            var response = await engine.GenerateAsync(new
            {
                context = "Payments are failing after today's deployment.",
                questions = new
                {
                    team = new
                    {
                        type = "string",
                        instructions = "Which team should handle this?",
                        @enum = new[] { "outage", "billing", "feature" },
                        permutations = 1
                    }
                }
            });
            Console.WriteLine(response.Result["team"].GetString());
        }
    }
}
```

`TurboJevEngine` implements both `IDisposable` and `IAsyncDisposable`. Existing
`await using` code remains valid; traditional `using` releases the same native
engine and its worker. Schema compilation, typed generation, evaluation,
probabilities, Dice, cancellation and timeouts use the same native ABI on all
three managed targets.

## Native deployment

The NuGet package bundles Windows x86, x64 and ARM64 DLLs. Modern .NET resolves
its runtime assets using the application's RID. Framework applications receive
the native directories through package build targets; the SDK loads the directory
matching the process architecture without changing its `PATH`.

Deploy the complete output, including `runtimes/win-*/native`, managed assemblies
and their dependencies. An x86 process needs x86 libraries regardless of the
host's architecture. Native ARM64 Framework execution needs Framework 4.8.1;
it has not been verified on the Windows x64 validation host. Native Linux/macOS
packages are not currently published.

## Validate the package

The release gate consumes a packed NuGet artifact instead of a project reference:

```powershell
pwsh -NoProfile -File scripts/smoke-dotnet-compat.ps1 `
  -PackagePath artifacts/nuget/TurboJev.VERSION.nupkg `
  -ModelPath path/to/model.gguf
```

It compiles the application targets above, executes the installed runtimes,
checks managed assembly selection, native loading, UTF-8, typed results, schema
helpers, cancellation, deadlines and synchronous/asynchronous disposal. A
non-pumping synchronization context catches accidental UI-context capture. The
gate also runs Framework x86. With `-ModelPath`, Framework 4.8, Core 3.1 and
.NET 8 load and evaluate a real GGUF; semantic model accuracy is a separate test.
Missing legacy runtimes appear as compile-only results, never execution passes.
Reports are written under `artifacts/dotnet-compat/`.

## Local validation: 2026-10-10

The locally packed `0.33.1-compat` test artifact passed all nine package-consumer
runs below. It was not uploaded to NuGet. The [saved report](validation/dotnet-compat.json)
includes the package SHA-256, selected SDK target, runtime and checked contracts.

| Consumer | Architecture | Contract checks | Real GGUF evaluation |
| --- | --- | --- | --- |
| Framework 4.8 | x64 | Passed | Passed |
| Framework 4.8 | x86 | Passed | Not run |
| Core 3.1 | x64 | Passed | Passed |
| .NET 5, 6, 7 | x64 | Passed | Not run |
| .NET 8 | x64 | Passed | Passed |
| .NET 9, 10 | x64 | Passed | Not run |

The real model was Qwen3-0.6B Q4_K_M on CPU. Contract checks use fixture inference
to test SDK behavior deterministically. ARM64 execution was not tested.
