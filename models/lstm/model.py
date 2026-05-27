from __future__ import annotations

"""BiLSTM architecture for binary per-label classification.

build_lstm() and prep_text() are called directly from src.train.train_lstm().
Trainer.build_estimator() is intentionally not implemented — LSTM is Keras-based
and does not fit the sklearn Pipeline used by GenericTrainer.
"""


def build_lstm(vocab: int = 12000, emb_dim: int = 100, maxlen: int = 306):
    """Return a compiled BiLSTM model for binary classification (one label)."""
    from keras.models import Sequential
    from keras.layers import (
        Bidirectional,
        Dense,
        Dropout,
        Embedding,
        LSTM,
        SpatialDropout1D,
    )

    model = Sequential(
        [
            Embedding(vocab, emb_dim, input_length=maxlen),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def prep_text(
    train_texts: list[str],
    test_texts: list[str],
    vocab: int = 12000,
    maxlen: int = 306,
):
    """Tokenize and pad text sequences.

    Returns (X_train_padded, X_test_padded, tokenizer).
    """
    from tf_keras.preprocessing.sequence import pad_sequences
    from tf_keras.preprocessing.text import Tokenizer

    tok = Tokenizer(num_words=vocab, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    X_tr = pad_sequences(
        tok.texts_to_sequences(train_texts), maxlen=maxlen, padding="post"
    )
    X_te = pad_sequences(
        tok.texts_to_sequences(test_texts), maxlen=maxlen, padding="post"
    )
    return X_tr, X_te, tok


class Trainer:
    def build_estimator(self, params: dict):
        raise NotImplementedError(
            "LSTM training is Keras-based. Call src.train.train_lstm() directly."
        )
