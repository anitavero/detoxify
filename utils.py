import sys


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

        Args
            filepath: the path to the file to be overwritten.

        Returns
            True if we can proceed with overwrite, False otherwise.
    """
    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    overwrite = get_input('[WARNING] %s already exist - overwrite? '
                          '[y/n]' % (filepath))
    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify --overwrite!')
    return True 
