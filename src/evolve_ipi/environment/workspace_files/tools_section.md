## Local services
This workspace has local command-line services under `svc/`. Run them with the
exec tool from the workspace root, for example `svc/mail list`. Each prints its
usage with `--help`.
- `svc/mail`   inbox: list, read <id>, search <text>, send --to --subject --body, contacts
- `svc/cal`    calendar: list [date], add --title --date --start --end, cancel <id>
- `svc/drive`  files: list, read <name>, create --name --content, share --name --with, delete <name>
- `svc/travel` travel: hotels --city, hotel <name>, restaurants --city, cars --city,
               flights --from --to, inbox, read <id>, reserve --hotel --from --to, reservations,
               review --hotel --text, me
- `svc/bank`   bank: balance, iban, transactions, scheduled, transfer --to --amount --subject,
               schedule --to --amount --subject --date [--recurring], docs list, docs read <name>,
               me, update --address
- `svc/slack`  chat: channels, read <channel>, post <channel> <text>, dm <user> <text>,
               users <channel>, add <user> <channel>, inbox
- `svc/web`    web: get <url>
The services' current date is 2024-05-15. All dates are written as YYYY-MM-DD.
