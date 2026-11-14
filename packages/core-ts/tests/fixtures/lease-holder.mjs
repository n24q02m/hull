// Another process sharing one pool entry: it takes the entry's lease and drops
// it again in a loop, through its own SQLite connection, the way a second hull
// process (another MCP client on the same `hull.db`) does. The statements match
// what `CredentialStore.leaseFirst` and `releaseLease` write.
//
// argv: <db path> <credential id> <pool clock now> <safety deadline ms>
// Prints `ready` once the connection is open; stops on SIGTERM or the deadline.
import Database from 'better-sqlite3'

const [dbPath, idText, nowText, deadlineText] = process.argv.slice(2)
const id = Number(idText)
const now = Number(nowText)
const deadline = Date.now() + Number(deadlineText)

const db = new Database(dbPath)
db.pragma('busy_timeout = 5000')
const take = db.prepare('UPDATE provider_credentials SET lease_until = ? WHERE id = ? AND lease_until <= ?')
const drop = db.prepare('UPDATE provider_credentials SET lease_until = 0 WHERE id = ?')

let stop = false
process.on('SIGTERM', () => {
  stop = true
})

process.stdout.write('ready\n')

// Hold each lease for a random 0-100 microseconds, so the release lands at
// every point of the other process's lease attempt, including just after it.
function hold() {
  const until = process.hrtime.bigint() + BigInt(Math.floor(Math.random() * 100_000))
  while (process.hrtime.bigint() < until) {
    // spin: a timer cannot wait this briefly
  }
}

function cycle() {
  // A batch per tick so SIGTERM is still delivered between batches.
  for (let index = 0; index < 200; index += 1) {
    if (take.run(now + 60, id, now).changes === 1) {
      hold()
      drop.run(id)
    }
  }
  if (stop || Date.now() > deadline) {
    db.close()
    return
  }
  setImmediate(cycle)
}

cycle()
