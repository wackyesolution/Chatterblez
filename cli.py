# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import os
from pathlib import Path

def cli_main():
    parser = argparse.ArgumentParser(
        description="Chatterblez  CLI - Convert EPUB/PDF to Audiobook",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', help='Path to a single EPUB or PDF file')
    group.add_argument('--batch', '-b', help='Path to a folder containing EPUB/PDF files for batch processing')
    group.add_argument('--remove-silence', '-rs', help='Path to an M4B file to remove silence from')

    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--filterlist', help='Comma-separated list of chapter names to ignore (case-insensitive substring match)')
    parser.add_argument('--wav', help='Path to a WAV file for voice conditioning (audio prompt)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed (default: 1.0)')
    parser.add_argument('--cuda', default=True, help='Use GPU via Cuda in Torch if available', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.cuda:
        import torch.cuda
        if torch.cuda.is_available():
            logging.info('CUDA GPU available')
            torch.set_default_device('cuda')
        else:
            logging.info('CUDA GPU not available. Defaulting to CPU')

    from core import main, remove_silence_from_audio

    # Prepare ignore_list
    ignore_list = [s.strip() for s in args.filterlist.split(',')] if args.filterlist else None

    # Prepare audio prompt
    audio_prompt_wav = args.wav if args.wav else None

    # Prepare output folder
    output_folder = args.output


    # Prepare speed
    speed = args.speed

    if args.remove_silence:
        input_file = Path(args.remove_silence)
        if not input_file.is_file():
            logging.error(f"Input file does not exist: {input_file}")
            sys.exit(1)
        
        output_file = Path(output_folder) / f"{input_file.stem}"
        if output_file.suffix.lower() != '.m4b':
            output_file = output_file.with_suffix('.m4b')
        remove_silence_from_audio(str(input_file), str(output_file))
        sys.exit(0)

    # Batch mode
    if args.batch:
        folder = Path(args.batch)
        if not folder.is_dir():
            logging.error(f"Batch folder does not exist: {folder}")
            sys.exit(1)
        supported_exts = [".epub", ".pdf"]
        batch_files = [
            str(folder / f)
            for f in os.listdir(folder)
            if os.path.isfile(str(folder / f)) and os.path.splitext(f)[1].lower() in supported_exts
        ]
        if not batch_files:
            logging.error("No supported files (.epub, .pdf) found in the selected folder.")
            sys.exit(1)
        main(
            file_path=None,
            pick_manually=False,
            speed=speed,
            output_folder=output_folder,
            batch_files=batch_files,
            ignore_list=ignore_list,
            audio_prompt_wav=audio_prompt_wav
        )
    # Single file mode
    elif args.file:
        file_path = args.file
        if not os.path.isfile(file_path):
            logging.error(f"File does not exist: {file_path}")
            sys.exit(1)
        main(
            file_path=file_path,
            pick_manually=False,
            speed=speed,
            output_folder=output_folder,
            batch_files=None,
            ignore_list=ignore_list,
            audio_prompt_wav=audio_prompt_wav
        )

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )
    cli_main()
