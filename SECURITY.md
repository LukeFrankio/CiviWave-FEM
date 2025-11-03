# Security Policy for CiviWave-FEM

Thank you for helping keep CiviWave-FEM secure. We take security reports seriously and appreciate responsible disclosure.

Please read this policy before reporting a vulnerability so we can respond quickly and coordinate fixes.

## Supported Versions

- We support the code in the `main` branch and any tagged releases. If you are unsure whether your version is supported, please include git commit, branch or tag information in your report.

## Reporting a Vulnerability

Preferred (private) reporting channels:

1. GitHub Security Advisories (recommended): Create a private security advisory for the repository (recommended). If you are a maintainer or have access, use the repository's "Security" → "Advisories" UI to create a report.

2. Email: If you prefer email, send your report to: <frankioluke@gmail.com>

When reporting, please include as much of the following as possible:

- Affected component(s) and version(s)
- Clear description of the vulnerability and impact
- Steps to reproduce (PoC) and example inputs/commands
- Expected vs actual behavior
- Any suggested mitigations or fixes
- Your disclosure preference (public immediately vs coordinated disclosure)

Do NOT post sensitive details publicly (issues, PRs, or other public channels) before we have coordinated a fix.

## PGP/GPG (optional)

If you would like to encrypt sensitive details, please use our PGP key:

> -----BEGIN PGP PUBLIC KEY BLOCK-----
> (PLEASE ADD MAINTAINER PGP KEY HERE)
> -----END PGP PUBLIC KEY BLOCK-----

If you need us to publish a key or you prefer a secure channel, contact us via the GitHub Advisory UI (private) or the email above and we'll share a key or alternate secure channel.

## What to Expect (Response Timeline)

- We will acknowledge receipt within 48 hours.
- We aim to provide an initial triage and plan within 5 business days.
- For verified vulnerabilities we will coordinate a fix and public disclosure timeline; typical coordinated disclosure target is within 60–90 days depending on complexity and downstream impact.
- If immediate action is required (high/critical severity), we will prioritize accordingly and may release an emergency patch.

## Severity and CVE

- We use CVSS-style scoring to help prioritize. If appropriate, we will request a CVE for serious vulnerabilities.

## Disclosure Policy

- We follow coordinated disclosure. Please do not publicly disclose vulnerabilities before we have a chance to respond and provide a fix, unless you explicitly state a preference for immediate public disclosure in your report.

## Safe Harbor for Security Researchers

We support good-faith security research. If you follow this policy and provide information about potential issues, we will not pursue legal action against you for the reported activity, provided you:

- Act in good faith and avoid privacy violations or data exfiltration where possible
- Share findings privately and do not attempt to exploit or disclose the vulnerability publicly prior to remediation

If you have questions about the scope of safe-harbor, contact us via the reporting channels.

## Handling Sensitive Data

We will treat any data shared for triage as confidential, store it only as needed to remediate the issue, and delete it on request when it is no longer required.

## Mitigation and Fixes

- For each security report we will provide a remediation plan and, when practical, a code fix, test coverage, and release notes describing the vulnerability and fix.
- We will backport critical fixes to supported releases where feasible and practical.

## Contact & Maintainers

- Repository owner: LukeFrankio
- Preferred reporting mechanism: GitHub Security Advisory (recommended), or email <frankioluke@gmail.com>

## If you cannot reach us

If you cannot use the above channels, open a support request with GitHub or contact the repo owner via their GitHub profile.
