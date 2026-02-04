public class SneakyThrows {
    public static void main(String[] args) {
        // 这里没有 try-catch，也没有 throws，但能抛出受检异常
        throwUnchecked(new java.io.IOException("我出来了！"));
    }

    @SuppressWarnings("unchecked")
    private static <T extends Throwable> void throwUnchecked(Throwable t) throws T {
        throw (T) t;
    }
}