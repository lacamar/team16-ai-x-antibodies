# Your Nebius access

AI x Antibodies Hackathon Oslo, 14 November 2026.

This folder belongs to one team. The keys inside are yours, so do not pass them
to another team.

| What is here | |
| --- | --- |
| `credentials.private.json` | Your login, your API key, the keys to a private 10 GB bucket |
| `agent/` | Everything Claude Code needs. You do not have to read it |

You get an account on the Nebius Scientific AI Platform: about 60 ready-to-use models,
already hosted. Nothing to install, no GPU needed for them.

## Keep your keys safe

Your submission is a public GitHub repository. Before your first commit, in your
project folder:

```bash
echo "*.private.json" >> .gitignore
```

Never paste the key into a chat or a prompt. Point tools at the file instead, as
the commands below do.

Two folders appear once you start working, `results/` and `nebius-runs/`. They
hold your outputs and your run ids, no keys, so you can commit them. Keeping
them helps: you have to show where each score came from.

## Way 1: the browser, 2 minutes

Open the `workbench_url` from `credentials.private.json` and sign in with the
login and password in the same file.

- In chat, pick the model **GLM 5.3 Flash**. The default one is slow for this
- **Apps** shows the models you can use
- **Workspace** is your bucket. It already holds about 700 public example files,
  including starters for molecular dynamics

## Way 2: Claude Code, 5 minutes

You need Python 3 and Claude Code. From inside this folder:

```bash
bash agent/install.sh
export NEBIUS_CREDS="$PWD/credentials.private.json"
python3 ~/.claude/skills/nebius-scientific-ai/forge.py models
```

The last command lists the models you can call and how long each one takes.

Work inside this folder. The tools look for your keys here. If you want to work
somewhere else, put that `export` line in your shell profile first, or the tools
will say they cannot find a credentials file.

Now start `claude` here. It picks up the skills by itself. Try a real job:

```bash
curl -sL -o 1zvy.pdb https://files.rcsb.org/download/1ZVY.pdb
```

Then ask Claude: *fold chain A of 1zvy.pdb with boltz2 and tell me the
confidence.* The fold itself takes about 80 seconds. The whole answer takes three
to six minutes, because Claude also reads the schema, searches an alignment and
downloads the result. 1ZVY is a nanobody bound to lysozyme.

One rule for Claude: never ask it for the full model catalog. That answer is
about 930 KB and it fills the whole context.

## While you wait

The first call to a model is the slow one, because the model has to load. Sixteen
teams share this, so start long jobs early.

Every run prints an id like `op-1234abcd`. Keep it. You can close the laptop and
come back later:

```bash
F=~/.claude/skills/nebius-scientific-ai/forge.py
python3 "$F" runs                      # everything you started here
python3 "$F" wait op-1234abcd
python3 "$F" fetch op-1234abcd --dest results/run1
```

Never submit the same job twice. You will only wait twice.

## Five problems that cost people time

1. **A run looks stuck.** It is queued, not broken. Check with `forge.py status
   op-xxxx`. A first call can wait 15 minutes
2. **"private storage is temporarily unavailable".** A platform hiccup. Run
   `forge.py resume op-xxxx` and it picks the same run back up
3. **The model refuses your input.** Read `forge.py schema <model>` and fix the
   arguments. Do not try shapes at random
4. **Molecular dynamics says "missing required fields: request".** Add
   `--wrap-request`. All five engines need it
5. **Python on a Mac cannot verify the certificate.** Run
   `Install Certificates.command` from the Python folder in Applications

## Stuck?

Find a Nebius mentor at the event. If a job is involved, bring its run id
