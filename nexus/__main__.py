"""Command line: python -m nexus [run | doctor | settings | check-plugin FILE]"""

import asyncio
import os
import sys

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

USAGE = """usage: python -m nexus [command]

  run                  start the assistant (the default)
  doctor               check the setup and say in plain words what is wrong
  settings             list every setting with its value and what it does
  settings --markdown  the same as a Markdown table (docs/SETTINGS.md)
  check-plugin FILE    load one plugin file and check it follows the rules
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "run"

    if command in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    if command == "doctor":
        from .doctor import run_doctor

        return run_doctor()

    if command == "settings":
        from . import config
        from .doctor import import_everything

        import_everything()

        if "--markdown" in argv:
            print(config.settings_markdown())
        else:
            for line in config.describe_settings():
                print(line)

        return 0

    if command == "check-plugin":
        if len(argv) < 2:
            print(USAGE)
            return 2

        from .testing import check_plugin_file

        problems = check_plugin_file(argv[1])

        for problem in problems:
            print(f"PROBLEM  {problem}")

        if not problems:
            print("OK  the plugin loads and its tools follow the rules.")

        return 1 if problems else 0

    if command == "run":
        from . import engine
        from .parts import PartError

        try:
            asyncio.run(engine.run())
        except KeyboardInterrupt:
            print()
            print("Shutting down.")
        except PartError as exc:
            print()
            print(f"Nexus cannot start: {exc}")
            print("Run `python -m nexus doctor` for a full check.")
            return 1

        return 0

    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
