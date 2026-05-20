"""Sequence scanner for detecting image sequences in folders"""
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = ['exr', 'jpg', 'jpeg', 'png', 'dpx', 'tga']

# Sequence patterns for detecting frame numbers
SEQUENCE_PATTERNS = [
    re.compile(r'^(.+?)[\._](\d{4,5})\.([a-z0-9]+)$', re.IGNORECASE),  # file_0001.exr, file.1001.jpg
    re.compile(r'^(.+?)\.(####)\.([a-z0-9]+)$', re.IGNORECASE),  # file.####.exr
]


class SequenceInfo:
    """Information about a detected image sequence"""
    
    def __init__(self, base_name: str, format_ext: str, folder_path: str, 
                 frame_start: int, frame_end: int, frame_count: int,
                 file_paths: List[str], total_size: int = 0, plate_name: str = None):
        self.base_name = base_name
        self.format = format_ext.upper()
        self.folder_path = folder_path
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.frame_count = frame_count
        self.file_paths = sorted(file_paths)
        self.total_size = total_size
        self.display_name = base_name
        self.first_file_path = file_paths[0] if file_paths else ""
        self.last_file_path = file_paths[-1] if file_paths else ""
        # Extract plate name if not provided
        if plate_name is None and file_paths:
            shot_name = extract_shot_name(self.first_file_path)
            self.plate_name = extract_plate_name(self.first_file_path, shot_name)
        else:
            self.plate_name = plate_name or 'default'
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'base_name': self.base_name,
            'format': self.format,
            'folder_path': self.folder_path,
            'frame_start': self.frame_start,
            'frame_end': self.frame_end,
            'frame_count': self.frame_count,
            'file_paths': self.file_paths,
            'total_size': self.total_size,
            'display_name': self.display_name,
            'first_file_path': self.first_file_path,
            'last_file_path': self.last_file_path,
            'plate_name': self.plate_name,
        }
    
    def get_sequence_pattern(self) -> str:
        """Get sequence pattern for engine (e.g., file.%04d.exr)"""
        if not self.file_paths:
            return ""
        
        # Get first file to determine pattern
        first_file = Path(self.first_file_path)
        base_name = first_file.stem
        
        # Determine padding from frame number
        frame_str = str(self.frame_start)
        padding = len(frame_str)
        if padding < 4:
            padding = 4
        
        # Get extension
        ext = first_file.suffix[1:]  # Remove dot
        
        # Create pattern
        pattern = f"{self.base_name}.%0{padding}d.{ext}"
        full_pattern = os.path.join(self.folder_path, pattern)
        
        return full_pattern.replace('\\', '/')


def extract_shot_name(file_path: str) -> str:
    """Extract shot name from file path (e.g., TML306_024_030)"""
    parts = file_path.replace('\\', '/').split('/')
    # Look for pattern like TML306_024_030
    pattern = re.compile(r'([A-Z0-9]+_\d+_\d+)')
    for part in parts:
        match = pattern.match(part)
        if match:
            return match.group(1)
    # Fallback: use parent folder name
    if len(parts) >= 2:
        return parts[-2]
    return 'Unknown'


def sequence_row_id(shot_name: str, seq: dict) -> str:
    """Stable id for a (shot, sequence) row — shared by scanner UI and batch engine callbacks."""
    folder = os.path.normcase(os.path.normpath(seq.get("folder_path", "") or ""))
    base = seq.get("base_name", "") or ""
    fmt = (seq.get("format") or "").upper()
    return f"{shot_name}\x1e{folder}\x1e{base}\x1e{fmt}"


def extract_plate_name(file_path: str, shot_name: str) -> str:
    """Extract plate name from file path (e.g., ftg01, ftg02)
    
    Args:
        file_path: Full file path
        shot_name: The shot name to help identify plate location
        
    Returns:
        Plate name or 'default' if not found
    """
    parts = file_path.replace('\\', '/').split('/')
    
    # Find shot name index
    shot_idx = -1
    for i, part in enumerate(parts):
        if shot_name in part:
            shot_idx = i
            break
    
    # Plate should be the folder immediately after shot folder
    if shot_idx >= 0 and shot_idx + 1 < len(parts):
        plate_name = parts[shot_idx + 1]
        # Skip common subfolders that aren't plates
        if plate_name.lower() not in ['exr', 'jpg', 'jpeg', 'png', 'dpx', 'tga', 'mov', 'mp4', 'avi']:
            return plate_name
    
    # Fallback: try to find any folder that looks like a plate name (alphanumeric)
    for part in parts:
        if re.match(r'^[a-z0-9]+$', part, re.IGNORECASE) and part.lower() not in ['exr', 'jpg', 'jpeg', 'png', 'dpx', 'tga']:
            return part
    
    return 'default'


def detect_sequences_in_folder(folder_path: str) -> List[SequenceInfo]:
    """
    Scan a folder recursively and detect image sequences
    
    Args:
        folder_path: Path to folder to scan
        
    Returns:
        List of SequenceInfo objects
    """
    sequences_map = defaultdict(lambda: {
        'frames': [],
        'paths': [],
        'sizes': [],
        'ext': '',
        'folder': ''
    })
    
    # Walk through all files
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            # Check if file matches sequence pattern
            matched = False
            for pattern in SEQUENCE_PATTERNS:
                match = pattern.match(file_name)
                if match:
                    base_name = match.group(1)
                    frame_str = match.group(2)
                    ext = match.group(3).lower()
                    
                    # Skip if not a supported format
                    if ext not in SUPPORTED_FORMATS:
                        continue
                    
                    # Parse frame number
                    try:
                        frame_num = int(frame_str)
                    except ValueError:
                        continue
                    
                    # Create key for grouping sequences
                    folder = os.path.dirname(file_path)
                    key = f"{folder}|{base_name}.{ext}"
                    
                    sequences_map[key]['frames'].append(frame_num)
                    sequences_map[key]['paths'].append(file_path)
                    sequences_map[key]['ext'] = ext
                    sequences_map[key]['folder'] = folder
                    
                    # Get file size
                    try:
                        size = os.path.getsize(file_path)
                        sequences_map[key]['sizes'].append(size)
                    except OSError:
                        sequences_map[key]['sizes'].append(0)
                    
                    matched = True
                    break
            
            if not matched:
                # Check for video files (single file sequences)
                ext = os.path.splitext(file_name)[1][1:].lower()
                if ext in ['mov', 'mp4', 'avi']:
                    folder = os.path.dirname(file_path)
                    base_name = os.path.splitext(file_name)[0]
                    key = f"{folder}|{base_name}.{ext}"
                    
                    sequences_map[key]['frames'].append(1)
                    sequences_map[key]['paths'].append(file_path)
                    sequences_map[key]['ext'] = ext
                    sequences_map[key]['folder'] = folder
                    
                    try:
                        size = os.path.getsize(file_path)
                        sequences_map[key]['sizes'].append(size)
                    except OSError:
                        sequences_map[key]['sizes'].append(0)
    
    # Convert to SequenceInfo objects
    sequences = []
    for key, seq_data in sequences_map.items():
        if len(seq_data['frames']) < 1:
            continue
        
        ext = seq_data['ext']
        is_video = ext in ['mov', 'mp4', 'avi']
        
        # For video files, treat as single file
        if is_video and len(seq_data['frames']) == 1:
            base_name = key.split('|')[1].split('.')[0]
            first_path = seq_data['paths'][0] if seq_data['paths'] else ""
            shot_name = extract_shot_name(first_path)
            plate_name = extract_plate_name(first_path, shot_name)
            sequences.append(SequenceInfo(
                base_name=base_name,
                format_ext=ext.upper(),
                folder_path=seq_data['folder'],
                frame_start=1,
                frame_end=1,
                frame_count=1,
                file_paths=seq_data['paths'],
                total_size=sum(seq_data['sizes']),
                plate_name=plate_name
            ))
            continue
        
        # For image sequences, require at least 2 frames
        if len(seq_data['frames']) < 2:
            continue
        
        frames_sorted = sorted(seq_data['frames'])
        frame_start = frames_sorted[0]
        frame_end = frames_sorted[-1]
        frame_count = len(frames_sorted)
        
        # Sort paths by frame number
        frame_path_pairs = list(zip(seq_data['frames'], seq_data['paths'], seq_data['sizes']))
        frame_path_pairs.sort(key=lambda x: x[0])
        sorted_paths = [p for _, p, _ in frame_path_pairs]
        sorted_sizes = [s for _, _, s in frame_path_pairs]
        
        base_name = key.split('|')[1].split('.')[0]
        
        # Extract plate name from first file path
        first_path = sorted_paths[0] if sorted_paths else ""
        shot_name = extract_shot_name(first_path)
        plate_name = extract_plate_name(first_path, shot_name)
        
        sequences.append(SequenceInfo(
            base_name=base_name,
            format_ext=ext.upper(),
            folder_path=seq_data['folder'],
            frame_start=frame_start,
            frame_end=frame_end,
            frame_count=frame_count,
            file_paths=sorted_paths,
            total_size=sum(sorted_sizes),
            plate_name=plate_name
        ))
    
    return sequences


def group_sequences_by_shot(sequences: List[SequenceInfo]) -> Dict[str, List[SequenceInfo]]:
    """
    Group sequences by shot name
    
    Args:
        sequences: List of SequenceInfo objects
        
    Returns:
        Dictionary mapping shot names to lists of sequences
    """
    grouped = defaultdict(list)
    
    for seq in sequences:
        shot_name = extract_shot_name(seq.folder_path)
        grouped[shot_name].append(seq)
    
    return dict(grouped)


def scan_folder_for_shots(folder_path: str) -> Dict[str, List[Dict]]:
    """
    Scan folder recursively and return shots with their sequences
    
    Args:
        folder_path: Path to folder to scan
        
    Returns:
        Dictionary mapping shot names to lists of sequence dictionaries
    """
    logger.info(f"Scanning folder: {folder_path}")
    
    # Detect all sequences
    sequences = detect_sequences_in_folder(folder_path)
    logger.info(f"Found {len(sequences)} sequences")
    
    # Group by shot
    grouped = group_sequences_by_shot(sequences)
    logger.info(f"Found {len(grouped)} shots")
    
    # Convert to dictionaries
    result = {}
    for shot_name, shot_sequences in grouped.items():
        result[shot_name] = [seq.to_dict() for seq in shot_sequences]
    
    return result

