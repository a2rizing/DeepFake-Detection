"""Gait verification module for deepfake detection."""

from .data_structures import GaitSignature, VerificationResult
from .keypoint_extractor import KeypointExtractor
from .signature_builder import SignatureBuilder
from .gait_verifier import GaitVerifier

__all__ = ['GaitSignature', 'VerificationResult', 'KeypointExtractor', 'SignatureBuilder', 'GaitVerifier']
