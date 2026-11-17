"""Make the accounts file for a private deployment.

    python scripts/make_users.py people.txt

people.txt has one person per line, ``username role [operator]``, e.g.

    apoorv     observer
    examiner1  observer
    team_sl    operator  Starlink
    fed        federation

Writes two files next to people.txt, both git-ignored:
  users.json      for the server (APP_USERS_FILE): scrypt hashes, no passwords
  passwords.txt   username and a fresh random password per person, to hand out
Running it again gives everyone new passwords.
"""
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.backend.main import hash_password  # noqa: E402

ROLES = ("observer", "federation", "operator")


def main(path: str) -> None:
    src = Path(path)
    users, lines = {}, []
    for raw in src.read_text(encoding="utf-8").splitlines():
        parts = raw.split("#")[0].split()
        if not parts:
            continue
        name, role = parts[0].lower(), parts[1].lower()
        if role not in ROLES:
            raise SystemExit(f"{name}: role must be one of {ROLES}")
        user = {"role": role}
        if role == "operator":
            if len(parts) < 3:
                raise SystemExit(f"{name}: an operator account needs the operator's name")
            user["operator"] = parts[2]
        password = secrets.token_urlsafe(9)
        users[name] = {**user, "password_hash": hash_password(password)}
        lines.append(f"{name:16s} {password}   ({role}{' ' + user['operator'] if role == 'operator' else ''})")
    (src.parent / "users.json").write_text(json.dumps(users, indent=1), encoding="utf-8")
    (src.parent / "passwords.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(users)} accounts -> {src.parent / 'users.json'} and {src.parent / 'passwords.txt'}")


if __name__ == "__main__":
    main(sys.argv[1])
