"""Utility to copy example FAQs into data/support_docs for quick demos."""
from pathlib import Path

def main():
    docs = Path("data/support_docs"); docs.mkdir(parents=True, exist_ok=True)
    (docs/"faq.md").write_text("""
# FAQ
## Reset password
To reset your password, click *Forgot Password* on the login page and follow the email link.
""".strip())

if __name__ == "__main__":
    main()