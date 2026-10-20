"""The Kinesis plan stays a stream plus IAM. It does not deploy or benchmark."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AWS = ROOT / "infra" / "aws"

READER_ACTIONS = {
    "kinesisvideo:DescribeStream",
    "kinesisvideo:GetDataEndpoint",
    "kinesisvideo:GetHLSStreamingSessionURL",
}
PRODUCER_ACTIONS = {
    "kinesisvideo:DescribeStream",
    "kinesisvideo:GetDataEndpoint",
    "kinesisvideo:PutMedia",
}
FORBIDDEN = (
    "AWS::ECS::",
    "AWS::ECR::",
    "AWS::EC2::",
    "AdministratorAccess",
    "AWS_SECRET_ACCESS_KEY",
    "AKIA",
)


def _policy(name: str) -> dict:
    return json.loads((AWS / "iam" / name).read_text(encoding="utf-8"))


def _actions(document: dict) -> set[str]:
    found: set[str] = set()
    for statement in document["Statement"]:
        actions = statement["Action"]
        found.update(actions if isinstance(actions, list) else [actions])
    return found


def test_reader_policy_matches_the_existing_consumer() -> None:
    document = _policy("kinesis-reader-policy.json")
    statement = document["Statement"][0]
    assert _actions(document) == READER_ACTIONS
    assert statement["Resource"] == "arn:aws:kinesisvideo:ap-southeast-2:*:stream/aegis-rescue/*"
    assert statement["Effect"] == "Allow"


def test_producer_policy_can_only_publish_video() -> None:
    document = _policy("kinesis-producer-policy.json")
    assert _actions(document) == PRODUCER_ACTIONS
    assert "kinesisvideo:GetHLSStreamingSessionURL" not in _actions(document)


def test_bootstrap_policy_cannot_deploy_the_app() -> None:
    document = _policy("member5-bootstrap-policy.json")
    actions = _actions(document)
    assert "kinesisvideo:CreateStream" in actions
    assert "cloudformation:CreateStack" in actions
    blob = json.dumps(document)
    for denied in ("ecs:", "ecr:", "ec2:", "logs:", "s3:", *FORBIDDEN):
        assert denied not in blob
    resources = [
        resource
        for statement in document["Statement"]
        for resource in (
            statement["Resource"]
            if isinstance(statement["Resource"], list)
            else [statement["Resource"]]
        )
    ]
    assert "arn:aws:kinesisvideo:ap-southeast-2:*:stream/aegis-rescue/*" in resources
    assert "arn:aws:iam::*:policy/aegis-kinesis-*" in resources
    assert "arn:aws:cloudformation:ap-southeast-2:*:stack/aegis-kinesis/*" in resources
    assert "cloudformation:CreateChangeSet" in actions
    assert "cloudformation:ExecuteChangeSet" in actions


def test_stream_template_is_only_the_kinesis_plan() -> None:
    template = (AWS / "kinesis-stream.yaml").read_text(encoding="utf-8")
    assert "AWS::KinesisVideo::Stream" in template
    assert "AWS::IAM::ManagedPolicy" in template
    assert "MediaType: video/h264" in template
    assert "Default: aegis-rescue" in template
    assert "Default: 24" in template
    assert "MinValue: 1" in template
    assert "ap-southeast-2" in template
    for action in READER_ACTIONS | PRODUCER_ACTIONS:
        assert action in template
    for denied in FORBIDDEN:
        assert denied not in template
