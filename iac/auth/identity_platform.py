"""Firebase Identity Platform configuration.

Enables Identity Platform on the GCP project and configures:
  - Email/password sign-in
  - Authorized domains (frontend domain + localhost for dev)
"""

import pulumi_gcp as gcp


def configure_identity_platform(project: str, authorized_domains: list[str]):
    # Enable the Identity Toolkit API (required for Identity Platform)
    gcp.projects.Service(
        "identity-toolkit-api",
        project=project,
        service="identitytoolkit.googleapis.com",
        disable_on_destroy=False,
    )

    # Enable Firebase Management API (needed to manage Firebase project config)
    gcp.projects.Service(
        "firebase-api",
        project=project,
        service="firebase.googleapis.com",
        disable_on_destroy=False,
    )

    # Configure Identity Platform
    gcp.identityplatform.Config(
        "identity-platform-config",
        project=project,
        autodelete_anonymous_users=False,
        sign_in=gcp.identityplatform.ConfigSignInArgs(
            allow_duplicate_emails=False,
            email=gcp.identityplatform.ConfigSignInEmailArgs(
                enabled=True,
                password_required=True,
            ),
        ),
        authorized_domains=authorized_domains,
    )
