{
  # Keep this line accurate and one line long: `nix flake metadata` prints it,
  # and it is the first thing a cold agent learns about the repo.
  description = "ir_racecar -- OpenCV image-join / IR lane-detection experiments for an MIT-RACEcar-style robot (archived ROS1 catkin workspace). Run `nix flake show` for the command map.";

  # nixpkgs is the only input, on purpose.
  #
  # flake-utils would buy exactly one thing here -- eachDefaultSystem -- which is
  # the three-line genAttrs below. In exchange it costs a second lock node in
  # every repo (flake-utils transitively pulls `systems`), a second upstream that
  # can break, and a hardcoded system list this repo cannot edit. That list is
  # currently broken: it still contains x86_64-darwin, which now throws (see
  # `systems` below).
  #
  # nixos-unstable is the same channel the author's own NixOS config tracks, so
  # `nix develop` here and `nixos-rebuild` there resolve the same store paths and
  # share one cache.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    # `...` rather than a closed { self, nixpkgs }: adding a second input later
    # would otherwise fail with "called with unexpected argument 'self'".
    { nixpkgs, ... }:
    let
      lib = nixpkgs.lib;

      # x86_64-darwin is deliberately absent. nixpkgs 26.11 replaced that whole
      # attribute set with `throw "Nixpkgs 26.11 has dropped support for
      # x86_64-darwin"`. genAttrs is lazy, so plain `nix develop` on Linux would
      # not notice -- it detonates later, on `nix flake check --all-systems`.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      # Stand-in for flake-utils.lib.eachDefaultSystem. Passes `pkgs` rather than
      # a system string, because that is what every call site below wants.
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # ======================================================================
      # PER-REPO BLOCK 1 -- the toolchain
      # ======================================================================
      # This repo ships no manifest at all -- no requirements.txt, no setup.py,
      # no CI. The dependency set below was derived by grepping the imports of
      # every .py under catkin_ws/src/camera_tests/scripts: cv2, numpy, scipy,
      # serial, yaml, tkinter (plus rospy/cv_bridge/sensor_msgs -- see the ROS
      # note further down).
      #
      # Because every one of those exists in nixpkgs, the interpreter is built
      # with withPackages instead of bootstrapping a .venv. Consequence, and it
      # is the whole point: this shell needs no network and there is no `setup`
      # verb to forget to run. Do NOT "improve" this by adding uv and a
      # requirements.txt -- there is no upstream pin to be faithful to, so a
      # generated requirements.txt would be a new source of truth nobody wrote.
      #
      # Pinned python313, not python3: the scripts were written for the 3.8 that
      # ROS Noetic shipped (the committed .pyc files are cpython-38), and a
      # rolling alias walking on to 3.14 would move this shell under the next
      # agent for no reason.
      #
      # GAP 1 -- ROS. camera_join.py, publish_camera_info.py and the
      # to_be_deleted/image_stitching_ROS* scripts import rospy, roslib,
      # cv_bridge and sensor_msgs, and catkin_ws/src/camera_tests/CMakeLists.txt
      # is a catkin package. ROS 1 Noetic is EOL and is NOT packaged in nixpkgs;
      # it lives in the out-of-tree nix-ros-overlay, which would be a second
      # input for a workspace that cannot be built anyway. So `catkin_make` and
      # the roslaunch flow in scripts/ros_start.batch are explicitly NOT covered
      # here, and there is no `build` verb pretending otherwise. Everything the
      # non-ROS scripts need IS covered.
      #
      # GAP 2 -- OpenCV highgui. nixpkgs builds opencv4 with `enableGtk3 ? false`,
      # so cv2.imshow / namedWindow / waitKey raise "The function is not
      # implemented. Rebuild the library with ... GTK+ 2.x support". That hits the
      # windowed demos (camera_test_no_ROS.py, showCamNoGUI.py, showFPS.py). The
      # fix is `pkgs.python313Packages.opencv4.override { enableGtk3 = true; }`,
      # which no binary cache has and which therefore costs a full ~30 min source
      # build of OpenCV for every cold agent. Left off on purpose: the same
      # scripts also need two physical USB cameras on /dev/video0 and /dev/video2
      # and an X/Wayland display, so a headless agent gains nothing from the
      # rebuild. Turn it on locally if you actually have the hardware in front of
      # you.
      #
      # Not covered and not coverable: scripts/pwm_Gen*.ino (Arduino sketches for
      # the external IR PWM generator) and cad/*.kicad_* -- those need the Arduino
      # IDE and KiCad plus real hardware, not a dev shell.
      #
      # Explicit `pkgs.foo`, never `with pkgs; [ ... ]`: when an attr disappears
      # in a nixpkgs bump, `with` reports a bare undefined identifier with no hint
      # of which set it came from, and the name is not greppable.
      toolchain = pkgs: [
        # ---- this repo's ecosystem ----
        (pkgs.python313.withPackages (ps: [
          ps.numpy
          ps.opencv4
          ps.pyserial
          ps.pyyaml
          ps.scipy
          # tkinter is a separate derivation in nixpkgs, not part of the base
          # interpreter. showCamNoGUI.py and toSerialApp.py do `from tkinter
          # import *`, so importing the interpreter alone would fail there.
          ps.tkinter
        ]))
        pkgs.ruff

        # ---- present in every repo in the fleet ----
        pkgs.git
        pkgs.jq
        pkgs.gnumake
      ];

      # ======================================================================
      # PER-REPO BLOCK 2 -- libraries that get dlopened, not linked
      # ======================================================================
      # Nothing in the toolchain above needs this: nixpkgs' cv2, numpy and scipy
      # are linked against the store copies already. It is here for the next
      # agent, because a repo with no manifest is exactly the one where somebody
      # reaches for `python -m venv .venv && .venv/bin/pip install <wheel>` --
      # and a manylinux wheel dlopens a libstdc++ that NixOS has no /usr/lib to
      # find it in. Keep the list at these two; LD_LIBRARY_PATH is a blunt
      # instrument.
      #
      # This fixes shared libraries only. A prebuilt *executable* out of a wheel
      # still needs a real ELF interpreter at the FHS path
      # `/lib64/ld-linux-x86-64.so.2`. That is a host setting (`environment.ldso`
      # / `programs.nix-ld.enable`) and no project flake can supply it.
      nativeLibs = pkgs: [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];

      # ======================================================================
      # PER-REPO BLOCK 3 -- constant environment variables
      # ======================================================================
      # Only values that are constants belong here. Anything that must READ an
      # existing value (LD_LIBRARY_PATH), UNSET something (SOURCE_DATE_EPOCH) or
      # touch the work tree goes in the shellHook further down.
      #
      # This attrset is applied to BOTH surfaces -- the dev shell and every
      # `nix run` wrapper -- so a command cannot behave differently depending on
      # how it was invoked.
      envVars = pkgs: {
        # The scripts/ directory has cpython-38 .pyc files committed into it from
        # the ROS Noetic days. Writing 3.13 ones beside them just dirties the
        # tree, and a dirty tree makes every nix call print "Git tree is dirty".
        PYTHONDONTWRITEBYTECODE = "1";
        PIP_DISABLE_PIP_VERSION_CHECK = "1";
      };

      # ======================================================================
      # PER-REPO BLOCK 4 -- the command map
      # ======================================================================
      # THE single source of truth. It generates `apps` (so `nix run .#lint`
      # works), the `dev-*` wrappers on PATH inside the shell, and `dev-help`.
      # Nothing is written twice, so `nix flake show` can never disagree with
      # what `dev-lint` actually runs.
      #
      # The house vocabulary is setup/build/test/lint/fmt/run and a verb the repo
      # has no meaning for is OMITTED rather than stubbed, because absence is
      # information. Missing here, and why:
      #   setup  nothing to install -- the interpreter above is complete offline
      #   build  the only buildable artifact is the catkin package, and that
      #          needs ROS 1 Noetic (GAP 1). `catkin_make` cannot run in here.
      #   test   there is no test suite in this repo. Not one test file, no CI
      #          workflow, and CMakeLists.txt leaves catkin_add_nosetests
      #          commented out. A `test` verb here would be a lie.
      #
      # `text` is bash under `set -euo pipefail`, shellcheck'd at BUILD time, and
      # it runs in the caller's current directory so an agent can test
      # uncommitted edits.
      commands = pkgs: {
        lint = {
          # Heads up for the next agent: this repo does NOT currently pass. The
          # scripts are 2022-era lab code and ruff's default rule set (E4/E7/E9/F)
          # flags bare `except:` (E722), star imports (F403/F405) and unused
          # imports throughout. That non-zero exit is the honest state of the
          # tree, not a broken flake -- do not chase it by weakening the verb.
          description = "ruff check (fails today -- pre-existing issues, see flake comment)";
          text = ''ruff check "$@"'';
        };
        fmt = {
          description = "ruff format (rewrites files)";
          text = ''ruff format "$@"'';
        };
        run = {
          # A bare `python3` is correct here and is NOT the mistake the house
          # style warns about: that warning is about repos whose deps live in a
          # .venv, where the wrappers' PATH prepend would shadow it with the store
          # interpreter. Here the store interpreter IS the one carrying cv2 and
          # numpy, so the prepend is doing the right thing. There is no .venv.
          #
          # Absolute script path, not a relative one, so this behaves the same
          # from any subdirectory. Python puts the script's own directory on
          # sys.path[0], which is what makes its `import image_join` resolve.
          #
          # Verified reaching the repo's own code and then failing on the repo's
          # own bug: camera_test_no_ROS.py calls
          # `ImageJoinFactory.create_instance(joinType=2, ...)` with keywords, but
          # image_join.py declares `create_instance(dict)` taking one positional
          # dict, so it dies with "unexpected keyword argument 'joinType'". That
          # is a pre-existing defect in this archived repo, not a toolchain
          # problem -- it fails fast rather than hanging, which is what matters
          # for an unattended agent.
          description = "start the non-ROS two-camera join demo (needs 2 USB cameras + a display; hits a pre-existing repo bug, see flake comment)";
          text = ''python3 "$REPO_ROOT/catkin_ws/src/camera_tests/scripts/camera_test_no_ROS.py" "$@"'';
        };
      };

      # ======================================================================
      # GENERIC MACHINERY -- byte-identical in all 41 repos, do not edit
      # ======================================================================

      # Prepend, never assign: a host LD_LIBRARY_PATH may be carrying something
      # the user needs, and clobbering it breaks binaries they launch from here.
      # Linux only -- on darwin the loader variable is DYLD_*, and exporting a
      # Linux-shaped value there is at best useless.
      ldPreamble =
        pkgs:
        lib.optionalString (pkgs.stdenv.hostPlatform.isLinux && nativeLibs pkgs != [ ]) ''
          export LD_LIBRARY_PATH="${lib.makeLibraryPath (nativeLibs pkgs)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';

      # Every command gets $REPO_ROOT. `nix run` and `nix develop` both start in
      # whatever directory they were invoked from, so a bare relative path
      # silently resolves against wherever the agent happened to be standing.
      # Note we do NOT cd there: commands act on the caller's cwd on purpose.
      rootPreamble = ''
        REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
        export REPO_ROOT
      '';

      # One derivation per command, reused by both `apps` and the dev shell, so
      # the two can never diverge. `dev-` prefixed because a bare `test` binary
      # earlier on PATH would shadow the POSIX shell builtin and quietly break
      # every script in the repo that uses it.
      wrappers =
        pkgs:
        lib.mapAttrs (
          name: cmd:
          pkgs.writeShellApplication {
            name = "dev-${name}";
            runtimeInputs = toolchain pkgs;
            runtimeEnv = envVars pkgs;
            meta.description = cmd.description;
            text = ''
              ${rootPreamble}
              ${ldPreamble pkgs}
              ${cmd.text}
            '';
          }
        ) (commands pkgs);

      helpFor =
        pkgs:
        let
          cmds = commands pkgs;
          names = lib.attrNames cmds;
          width = lib.foldl' (a: n: lib.max a (builtins.stringLength n)) 0 names;
          pad = n: n + lib.concatStrings (lib.genList (_: " ") (width - builtins.stringLength n));
          line = n: c: "  dev-${pad n}  ${c.description}";
        in
        pkgs.writeShellApplication {
          name = "dev-help";
          meta.description = "print this repo's command map (works offline)";
          text = ''
            cat <<'EOF'
            ${lib.concatStringsSep "\n" (lib.mapAttrsToList line cmds)}
            EOF
          '';
        };
    in
    {
      # `nix flake show` -- the discovery entrypoint, and deliberately the whole
      # machine-facing contract: every app carries a meta.description, which
      # `nix flake show` prints inline and `nix flake show --json` exposes at
      # .apps.<system>.<name>.description. Pure evaluation, so an agent gets the
      # entire command map in one cheap call without reading a README.
      #
      # Do NOT invent a top-level output for this (`agentManifest`, `probeThing`
      # ...). Nix answers with `warning: unknown flake output '<name>'` on every
      # single `nix flake check`, forever.
      apps = forAllSystems (
        pkgs:
        lib.mapAttrs (name: cmd: {
          type = "app";
          program = "${(wrappers pkgs).${name}}/bin/dev-${name}";
          meta.description = cmd.description;
        }) (commands pkgs)
      );

      # `nix develop` -- the toolchain, plus a dev-<verb> for every app.
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = toolchain pkgs ++ lib.attrValues (wrappers pkgs) ++ [ (helpFor pkgs) ];

          env = envVars pkgs;

          # Some C extensions compile at -O0, where glibc's _FORTIFY_SOURCE
          # becomes a hard error instead of a warning.
          hardeningDisable = [ "fortify" ];

          shellHook = ''
            # mkShell inherits SOURCE_DATE_EPOCH=315532800 (1980-01-01) from
            # stdenv, and any wheel or zip built in here then dies with "ZIP does
            # not support timestamps before 1980".
            unset SOURCE_DATE_EPOCH

            ${rootPreamble}
            ${ldPreamble pkgs}

            # Nothing networked, nothing stateful and nothing interactive above
            # this line, and nothing below it either. No venv creation, no
            # `pip install`. Bootstrapping in the hook makes a cold
            # `nix develop -c python3 ...` start downloading before it runs
            # anything, on EVERY invocation -- the exact failure an unattended
            # agent cannot diagnose.

            # The banner is interactive-only, and this guard is load-bearing:
            # shellHook output lands on the STDOUT of `nix develop -c <cmd>`, so
            # an unguarded echo corrupts anything parsing it
            # (`nix develop -c cat x.json | jq` fails to parse). $- is the only
            # reliable discriminator here -- it lacks `i` for `nix develop -c`
            # and has it at an interactive prompt. Do not test $PS1 (unset in
            # both) or $IN_NIX_SHELL (set in both). >&2 is the second layer, for
            # the case where a caller runs us on a pty.
            case $- in
              *i*) echo "ir_racecar dev shell -- 'dev-help' for the command map" >&2 ;;
            esac
          '';
        };
      });

      # `nix flake check` -- honest by construction. It realises the toolchain
      # closure (so a typo'd or currently-broken attr fails here) and builds
      # every wrapper, which runs shellcheck over every command text. It also
      # imports the modules this repo actually needs, which is the cheap real
      # check available in a repo with no test suite: it would catch a nixpkgs
      # bump that drops cv2's python bindings or moves tkinter. NEVER add a check
      # that always passes: an agent reads "all checks passed!" as a signal.
      checks = forAllSystems (pkgs: {
        toolchain =
          pkgs.runCommand "toolchain-check"
            {
              nativeBuildInputs = toolchain pkgs ++ lib.attrValues (wrappers pkgs);
            }
            ''
              for verb in ${lib.escapeShellArgs (lib.attrNames (commands pkgs))}; do
                command -v "dev-$verb" > /dev/null || {
                  echo "dev-$verb is not on PATH" >&2
                  exit 1
                }
              done
              # tkinter is imported without touching Tk itself -- there is no
              # display in a build sandbox, so `tkinter.Tk()` would fail here for
              # a reason that has nothing to do with the toolchain.
              python3 -c 'import cv2, numpy, scipy, serial, tkinter, yaml; print(cv2.__version__)'
              touch "$out"
            '';
      });

      # `nix fmt` -- formats the *Nix* in this repo; project code is `dev-fmt`.
      # nixfmt-tree (the treefmt wrapper) rather than bare nixfmt, because bare
      # nixfmt tries to parse every path handed to it and fails on non-Nix files.
      # This file ships already formatted, so `nix fmt` is a no-op rather than a
      # diff.
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
