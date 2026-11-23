// Load before the Command Code ESM bundle so every child process enters Landlock.
const child = require('node:child_process');
const { syncBuiltinESMExports } = require('node:module');
const path = require('node:path');

const { spawn, spawnSync, execFile, execFileSync } = child;
const launcher = path.join(__dirname, 'command-code-workspace');
const workspace = process.env.AW_COMMAND_CODE_WORKSPACE;
if (!workspace) throw new Error('Command Code workspace missing');

function invocation(file, args, options) {
    options ??= {};
    const env = { PATH: process.env.PATH, AW_COMMAND_CODE_WORKSPACE: workspace };
    for (const [key, value] of Object.entries(process.env)) {
        if (key === 'LANG' || key === 'TERM' || key === 'TZ' || key.startsWith('LC_')) env[key] = value;
    }
    const command = options.shell
        ? [typeof options.shell === 'string' ? options.shell : '/bin/sh', '-c', [file, ...args].join(' ')]
        : [file, ...args];
    return [launcher, command, { ...options, shell: false, detached: false, env }];
}

function parameters(args, options, callback) {
    if (typeof args === 'function') return [[], {}, args];
    if (!Array.isArray(args)) return [[], args, typeof options === 'function' ? options : callback];
    if (typeof options === 'function') return [args, {}, options];
    return [args, options, callback];
}

child.spawn = (file, args, options) => {
    const [argv, opts] = parameters(args, options);
    return spawn(...invocation(file, argv, opts));
};
child.spawnSync = (file, args, options) => {
    const [argv, opts] = parameters(args, options);
    return spawnSync(...invocation(file, argv, opts));
};
child.execFile = (file, args, options, callback) => {
    const [argv, opts, done] = parameters(args, options, callback);
    return execFile(...invocation(file, argv, opts), done);
};
child.execFileSync = (file, args, options) => {
    const [argv, opts] = parameters(args, options);
    return execFileSync(...invocation(file, argv, opts));
};
child.exec = (command, options, callback) => {
    const [, opts, done] = parameters(options, callback);
    return child.execFile(command, [], { ...opts, shell: opts?.shell ?? true }, done);
};
child.execSync = (command, options) => child.execFileSync(command, [], { ...options, shell: options?.shell ?? true });
child.fork = () => { throw new Error('Command Code workspace does not expose Node IPC'); };
syncBuiltinESMExports();
