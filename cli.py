# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import os
import time
from pathlib import Path

def cli_main():
    start_time = time.time() # Start timer

    parser = argparse.ArgumentParser(
        description="Chatterblez  CLI - Convert EPUB/PDF to Audiobook",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', help='Path to a single EPUB or PDF file')
    group.add_argument('--batch', '-b', help='Path to a folder containing EPUB/PDF files for batch processing')

    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--filterlist', help='Comma-separated list of chapter names to ignore (case-insensitive substring match)')
    parser.add_argument('--wav', help='Path to a WAV file for voice conditioning (audio prompt)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed (default: 1.0)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                        help='Select compute device (default: auto).')
    parser.add_argument('--tts-mode', choices=['standard', 'multilingual'], default='standard',
                        help='Choose between the default English model and the multilingual model.')
    parser.add_argument('--language-id', default='en',
                        help='Language code for multilingual synthesis (e.g. en, it, fr).')

    # Silence trimming parameters
    parser.add_argument('--enable-silence-trimming', action='store_true', help='Enable silence trimming on the generated audio chapters.')
    parser.add_argument('--silence-thresh', type=int, default=-50, help='The upper bound for what is considered silence in dBFS.')
    parser.add_argument('--min-silence-len', type=int, default=500, help='The minimum length of a silence in milliseconds.')
    parser.add_argument('--keep-silence', type=int, default=100, help='The amount of silence to leave at the beginning and end of the trimmed audio.')

    # Model parameters
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty (default: 1.1)')
    parser.add_argument('--min-p', type=float, default=0.02, help='Min P for sampling (default: 0.02)')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top P for sampling (default: 0.95)')
    parser.add_argument('--exaggeration', type=float, default=0.4, help='Exaggeration factor (default: 0.4)')
    parser.add_argument('--cfg-weight', type=float, default=0.8, help='CFG weight (default: 0.8)')
    parser.add_argument('--temperature', type=float, default=0.85, help='Temperature for sampling (default: 0.85)')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    from core import main, choose_device

    try:
        device = choose_device(args.device)
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)
    logging.info(f'Using device: {device}')
    if device in ('cuda', 'mps'):
        import torch
        torch.set_default_device(device)

    # Prepare ignore_list
    ignore_list = [s.strip() for s in args.filterlist.split(',')] if args.filterlist else None

    # Prepare audio prompt
    audio_prompt_wav = args.wav if args.wav else None

    # Prepare output folder
    output_folder = args.output


    # Prepare speed
    speed = args.speed

    # Batch mode
    if args.batch:
        folder = Path(args.batch)
        if not folder.is_dir():
            logging.error(f"Batch folder does not exist: {folder}")
            elapsed_time = time.time() - start_time
            logging.info(f"Script finished in {elapsed_time:.2f} seconds")
            sys.exit(1)
        supported_exts = [".epub", ".pdf"]
        batch_files = [
            str(folder / f)
            for f in os.listdir(folder)
            if os.path.isfile(str(folder / f)) and os.path.splitext(f)[1].lower() in supported_exts
        ]
        if not batch_files:
            logging.error("No supported files (.epub, .pdf) found in the selected folder.")
            elapsed_time = time.time() - start_time
            logging.info(f"Script finished in {elapsed_time:.2f} seconds")
            sys.exit(1)
        main(
            file_path=None,
            pick_manually=False,
            speed=speed,
            output_folder=output_folder,
            batch_files=batch_files,
            ignore_list=ignore_list,
            audio_prompt_wav=audio_prompt_wav,
            device=device,
            tts_mode=args.tts_mode,
            language_id=args.language_id,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_p=args.top_p,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            enable_silence_trimming=args.enable_silence_trimming,
            silence_thresh=args.silence_thresh,
            min_silence_len=args.min_silence_len,
            keep_silence=args.keep_silence
        )
    # Single file mode
    elif args.file:
        file_path = args.file
        if not os.path.isfile(file_path):
            logging.error(f"File does not exist: {file_path}")
            elapsed_time = time.time() - start_time
            logging.info(f"Script finished in {elapsed_time:.2f} seconds")
            sys.exit(1)
        main(
            file_path=file_path,
            pick_manually=False,
            speed=speed,
            output_folder=output_folder,
            batch_files=None,
            ignore_list=ignore_list,
            audio_prompt_wav=audio_prompt_wav,
            device=device,
            tts_mode=args.tts_mode,
            language_id=args.language_id,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_p=args.top_p,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            enable_silence_trimming=args.enable_silence_trimming,
            silence_thresh=args.silence_thresh,
            min_silence_len=args.min_silence_len,
            keep_silence=args.keep_silence
        )
    elapsed_time = time.time() - start_time
    logging.info(f"Script finished in {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('chatterbox').setLevel(logging.WARNING)
    cli_main()
