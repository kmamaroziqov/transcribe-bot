"""Prepare the local server before migrating this bot off Telegram's hosted API."""
import os
from pathlib import Path
import subprocess

from dotenv import dotenv_values, set_key
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    values = dotenv_values(ROOT / '.env')
    required = ('TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'TELEGRAM_BOT_TOKEN')
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise SystemExit('Add these to .env before activation: ' + ', '.join(missing))
    if not values['TELEGRAM_API_ID'].isdigit():
        raise SystemExit('TELEGRAM_API_ID must be a number.')
    env = os.environ.copy()
    if os.name == 'nt' and (ROOT / '.docker-build/config.json').exists():
        env['DOCKER_CONFIG'] = str(ROOT / '.docker-build')
    base = ['docker']
    if os.name == 'nt':
        base += ['--host', 'npipe:////./pipe/dockerDesktopLinuxEngine']
    old = base + ['compose', '--profile', 'telegram']
    local = base + ['compose', '-f', 'compose.yaml', '-f', 'compose.local-telegram.yaml', '--profile', 'telegram']
    def run(command):
        # Capture output so a dependency cannot accidentally display credentials.
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        output = result.stdout
        for key in required:
            output = output.replace(values[key], '[REDACTED]')
        print(output, flush=True)
        if result.returncode:
            raise RuntimeError('Docker step failed; inspect the redacted output above.')
    print('Building the application and starting the local Telegram server...', flush=True)
    run(local + ['build', 'stt'])
    run(local + ['up', '-d', '--wait', '--wait-timeout', '180', 'telegram-api'])
    run(old + ['stop', '-t', '7200', 'bot'])
    if values.get('TELEGRAM_API_URL') != 'http://telegram-api:8081':
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post('https://api.telegram.org/bot' + values['TELEGRAM_BOT_TOKEN'] + '/logOut')
            if response.status_code != 200 or not response.json().get('ok'):
                raise RuntimeError('Telegram did not confirm migration logout.')
        except Exception:
            run(old + ['up', '-d', '--no-deps', '--no-build', 'bot'])
            raise RuntimeError('Migration logout could not be confirmed; the original bot was restarted.') from None
    for key, value in {
        'TELEGRAM_API_URL': 'http://telegram-api:8081',
        'MAX_FILE_MB': '4096', 'STT_MAX_FILE_MB': '4096',
        'COMPOSE_PATH_SEPARATOR': '|',
        'COMPOSE_FILE': 'compose.yaml|compose.local-telegram.yaml',
    }.items():
        set_key(str(ROOT / '.env'), key, value)
    run(local + ['up', '-d', '--no-build', '--wait', '--wait-timeout', '180'])
    print('Local Telegram API activated with a 4096 MB file limit. Duration limit is unchanged.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Activation stopped ({type(exc).__name__}). Check the step above; credentials are not printed.', flush=True)
        raise SystemExit(1)
