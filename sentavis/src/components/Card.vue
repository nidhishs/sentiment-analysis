<template>
    <div class="card">
        <div class="card__info">
            <div class="icon">😶</div>
            <div class="name-handle">
                <div class="name"></div>
                <div class="handle"></div>
            </div>
        </div>
        <p class="card__text">
            {{ body }}
        </p>
    </div>
</template>

<script>
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min) + min);
}

export default {
    name: "Card",
    props: {
        body: String,
        sentiment: Number,
    },
    mounted() {
        const name = document.querySelector(".name");
        const handle = document.querySelector(".handle");
        const icon = document.querySelector(".icon");
        name.style.width = randomInt(15, 35) * 5 + "px";
        handle.style.width = randomInt(10, 15) * 5 + "px";
        if (this.sentiment >= 0.5) {
            icon.innerText = "🙂";
        } else {
            icon.innerText = "🙁";
        }
    },
};
</script>

<style scoped lang="scss">
.card {
    position: relative;
    width: 600px;
    padding: 20px;
    margin: 20px;
    border-radius: 15px;

    background: #f8f8f8;
    box-shadow: 5px 5px 5px #d8d8d8, -5px -5px 5px #ffffff;

    &__info {
        display: flex;
        flex-direction: row;
        align-items: center;
        margin-bottom: 15px;

        .icon {
            width: 40px;
            height: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 2.5rem;
        }

        .name-handle {
            margin-left: 10px;
            .name {
                background: #d5d5d5;
                height: 9px;
                margin-bottom: 5px;
            }
            .handle {
                background: #e4e4e4;
                height: 6px;
            }
        }
    }

    &__text {
        font-size: 14px;
        padding: 15px;
        background: #ffffff;
        box-shadow: inset 5px 5px 5px #d8d8d8, inset -5px -5px 5px #ffffff;
        border-radius: 15px;
    }
}
</style>
