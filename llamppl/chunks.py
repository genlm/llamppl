import asyncio
import string

from .modeling import submodel


@submodel
async def sample_word(self, context, max_tokens=5, allow_punctuation=True):
    """Sample a word from the `LMContext` object `context`."""
    last_token = (
        context.lm.str_vocab[context.tokens[-1]] if len(context.tokens) > 0 else ""
    )
    last_character = last_token[-1] if len(last_token) > 0 else ""
    needs_space = last_character not in string.whitespace and last_character not in [
        "-",
        "'",
        '"',
    ]
    if needs_space:
        starts_word_mask = context.lm.masks.STARTS_NEW_WORD
    else:
        starts_word_mask = context.lm.masks.CONTINUES_CURRENT_WORD

    # Force model to start a new word
    await self.observe(context.mask_dist(starts_word_mask), True)

    word = ""
    num_tokens = 0
    while True:
        token = await self.sample(context.next_token())
        word += context.lm.str_vocab[token.token_id]
        num_tokens += 1

        if num_tokens == max_tokens:
            await self.observe(
                context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD), False
            )
            break

        if not (
            await self.sample(
                context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD)
            )
        ):
            break

    # Sample punctuation, if desired
    punctuation = ""
    if allow_punctuation and await self.sample(
        context.mask_dist(context.lm.masks.PUNCTUATION)
    ):
        punctuation_token = await self.sample(context.next_token())
        punctuation = context.lm.str_vocab[punctuation_token.token_id]

    return word, punctuation


@submodel
async def sample_word_2(
    self,
    context,
    max_chars: int = None,
    allow_mid_punctuation: bool = True,
    allow_end_punctuation: bool = True,
):
    """Sample a word from the `LMContext` object `context`.

    Unlike sample_word() above, this method allows for character-level control over the length of the word.
    It also allows for control over the presence of punctuation in the middle and at the end of the word.

    Args:
        max_chars (int): Maximum number of characters in the word. If None, the model will sample a word of any length.
        allow_mid_punctuation (bool): If True, the model may sample punctuation in the middle of the word.
        allow_end_punctuation (bool): If True, the model may sample punctuation at the end of the word.

    Returns:
        Tuple[str, str]: The sampled word and punctuation
    """
    # NOTE: Yields control back to the event loop. Necessary to allow timeouts to work correctly when this method is called in a loop.
    await asyncio.sleep(0)

    # This approach sometimes breaks with max_chars = 1
    if max_chars is not None:
        assert max_chars > 1

    last_token = (
        context.lm.str_vocab[context.tokens[-1]] if len(context.tokens) > 0 else ""
    )
    last_character = last_token[-1] if len(last_token) > 0 else ""
    needs_space = last_character not in string.whitespace and last_character not in [
        "-",
        "'",
        '"',
    ]
    if needs_space:
        starts_word_mask = context.lm.masks.STARTS_NEW_WORD
    else:
        starts_word_mask = context.lm.masks.CONTINUES_CURRENT_WORD

    # Force model to start a new word
    await self.observe(context.mask_dist(starts_word_mask), True)

    word = ""
    while True:
        # Force model to sample a token with an appropriate number of characters
        if max_chars is not None:
            await self.observe(
                context.mask_dist(
                    context.lm.masks.MAX_TOKEN_LENGTH[max_chars - len(word.strip())]
                ),
                True,
            )

        token = await self.sample(context.next_token())
        word += context.lm.str_vocab[token.token_id]

        # If we ran out of chars, break
        if max_chars is not None and len(word.strip()) >= max_chars:
            await self.observe(
                context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD), False
            )
            break

        # If the model wants to end the word, break
        if not (
            await self.sample(
                context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD)
            )
        ):
            break

    # Sample punctuation, if desired
    mid_punctuation, end_punctuation = "", ""

    mask = set()
    if allow_mid_punctuation:
        mask = mask | context.lm.masks.MID_PUNCTUATION
    if allow_end_punctuation:
        mask = mask | context.lm.masks.END_PUNCTUATION

    if mask and await self.sample(context.mask_dist(mask)):
        token = await self.sample(context.next_token())
        if token.token_id in context.lm.masks.MID_PUNCTUATION:
            mid_punctuation = context.lm.str_vocab[token.token_id]
        if token.token_id in context.lm.masks.END_PUNCTUATION:
            end_punctuation = context.lm.str_vocab[token.token_id]

    return word, mid_punctuation, end_punctuation
